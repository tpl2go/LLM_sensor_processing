import logging
import re

import numpy as np
import openai

from constr_system_prompt import append_signal_knowledge, append_task_knowledge
from agent import OpenAIAgent, ReflectOpenAIAgent, EvalOpenAIAgent
from utils import (
    extract_code,
    write_to_csv_file,
    redirect_stdout,
    extract_array_from_str,
    convert_to_message,
    add_execution_string,
    add_gaurdrail_for_no_api,
)
from prompt import no_code_feedback, reflect_prompt, eval_prompt, eval_prompt_coding, verifier_prompt

logger = logging.getLogger(__name__)


def iteration_program_output(output):
    if "An error occurred:" in output:
        program_output = "The above program printed errors. Please fix it:\n" + output
    elif len(output) == 0:
        program_output = (
            "The above code completed successfully or no code is written. "
            "If this is meant to be the case, state the keyword [SUCCESS] and the iteration will stop."
        )
    elif len(output) >= 2048:
        program_output = "The above program printed too lengthy output. I've cropped it to 4096 characters for you.\n" + output[:4096]
    else:
        program_output = "The above program printed:\n" + output

    program_output = ">>>>>>" + program_output
    logger.info("Program output summary prepared | chars=%s", len(program_output))
    return program_output


def format_user_query(args):
    if args.query is not None:
        query_path = "query" + "/" + args.query + ".txt"
        logger.info("Loading query from file: %s", query_path)
        with open(query_path, encoding="utf-8") as file:
            user_message = file.readline()
    else:
        logger.info("No query file provided. Reading interactive user input.")
        user_message = input("user: ")

    if args.knowledge_signal:
        logger.info("Appending signal knowledge to query.")
        user_message += append_signal_knowledge(args)
    if args.knowledge_task:
        logger.info("Appending task knowledge to query.")
        user_message += append_task_knowledge(args)

    if args.mode != "text":
        user_message = "\\QUERY[{}]".format(user_message)

    logger.info("User query prepared | chars=%s", len(user_message))
    return user_message


def evaluating_output(args, input_array=None, write_result=False):
    logger.info(
        "Evaluating output | mode=%s output_file=%s target_file=%s write_result=%s",
        args.mode,
        args.output_file,
        args.target_file,
        write_result,
    )
    if args.target_file is not None:
        if args.mode == "text":
            from mse_distance import compute_mse_from_target

            mse = compute_mse_from_target(args, input_array)
        else:
            from mse_distance import compute_mse

            mse = compute_mse(args.output_file, args.target_file, args)

        if "speech" in args.target_file:
            target_metric = "SDR (speech to noise ratio)"
        elif "synthesis" in args.target_file:
            target_metric = "F1 score"
        else:
            target_metric = "MSE (mean square error)"

        m_mse = "The {} is: {:.4f}".format(target_metric, mse)
        logger.info("Evaluation complete | %s", m_mse)

        if args.write_to_csv and write_result:
            logger.info("Writing evaluation result to CSV.")
            write_to_csv_file(args.mode, args.query, args.index, args.log_name, mse)
        return m_mse

    logger.info("Ground truth missing. Returning fallback message.")
    return "The groundtruth is not provided."


def Agent_with_reflection(
    openai_key: str,
    system_prompt: str,
    global_dict,
    local_dict,
    model="gpt-3.5-turbo-0613",
    temperature=0.2,
    top_p=0.1,
    args=None,
):
    logger.info(
        "Agent_with_reflection start | model=%s mode=%s eval=%s num_trial=%s",
        model,
        args.mode,
        args.eval,
        args.num_trial,
    )
    openai.api_key = openai_key
    if "Llama" in args.openai or "Qwen" in args.openai:
        openai.api_base = args.base_url
        logger.info("Configured Together-compatible OpenAI base_url=%s", args.base_url)

    n = args.num_trial
    reflection_piece = None
    reflect_llm = ReflectOpenAIAgent(args, model=model, system_prompt=reflect_prompt, temperature=1, top_p=1)

    if args.eval == "self_vis":
        eval_llm = EvalOpenAIAgent(args, model=model, system_prompt=eval_prompt, temperature=1, top_p=1)
    elif args.eval == "self_coding":
        eval_llm = EvalOpenAIAgent(args, model=model, system_prompt=eval_prompt_coding, temperature=1, top_p=1)
    elif args.eval == "self_verifier":
        eval_llm = EvalOpenAIAgent(args, model=model, system_prompt=verifier_prompt, temperature=1, top_p=1)
    else:
        eval_llm = None

    succeed = False
    performance_list = []
    for _trial in range(n):
        logger.info("========> Round %s starts...", _trial + 1)

        reply, chat, user_message, m_mse = Agent_with_API(
            openai_key,
            system_prompt,
            global_dict,
            local_dict,
            _trial,
            model,
            temperature,
            top_p,
            args,
            reflection_piece=reflection_piece,
            write_result=False,
        )
        logger.info("Round %s completed | metric=%s", _trial + 1, m_mse)

        performance_list.append(m_mse)

        if n >= 2 and _trial <= n - 1:
            if args.eval in ("self_vis", "self_verifier", "self_coding"):
                logger.info("Running evaluator for trial=%s", _trial)
                eval_result = eval_llm.eval(context=chat, question=user_message, global_dict=global_dict, local_dict=local_dict, trial=_trial)
                reflect_llm.update(context=chat, question=user_message, performance=eval_result)
                logger.info("Evaluator returned | chars=%s", len(eval_result))

                if "The test passed." in eval_result:
                    succeed = True
                    logger.info("Evaluator indicates success. Stopping reflection loop early.")
                else:
                    logger.info("Evaluator indicates failure. Requesting reflection.")
                    reflection_piece = reflect_llm.step(trial=_trial)
                    reflect_llm.reset()

            elif args.eval == "env":
                feedback = convert_to_message(m_mse)
                logger.info("Environment feedback: %s", feedback)
                reflect_llm.update(context=chat, question=user_message, performance=feedback)

                reflection_piece = reflect_llm.step(trial=_trial)
                reflect_llm.reset()

                if "[SUCCESS]" in reflection_piece:
                    succeed = True
                    logger.info("Reflection returned [SUCCESS].")

        if _trial == n - 1 or succeed:
            mse = m_mse.split("is: ")[-1]
            if not succeed:
                mse = performance_list[0].split("is: ")[-1]
            if args.write_to_csv:
                logger.info("Writing final result to CSV | score=%s", mse)
                write_to_csv_file(args.mode, args.query, args.index, args.log_name, mse)
            logger.info("Agent_with_reflection exiting after trial=%s", _trial)
            return reply


def Agent_with_API(
    openai_key: str,
    system_prompt: str,
    global_dict,
    local_dict,
    trial,
    model="gpt-3.5-turbo-0613",
    temperature=0.2,
    top_p=0.1,
    args=None,
    reflection_piece=None,
    write_result=True,
):
    logger.info("Agent_with_API start | trial=%s temperature=%s top_p=%s", trial, temperature, top_p)
    openai.api_key = openai_key
    if "Llama" in args.openai:
        openai.api_base = args.base_url
        logger.info("Configured Together-compatible OpenAI base_url=%s", args.base_url)

    agent = OpenAIAgent(args, model=model, system_prompt=system_prompt, temperature=temperature, top_p=top_p)

    logger.info("Running warmup assistant step before sending query.")
    reply = agent.step()

    user_message = format_user_query(args)
    agent.update(content=user_message, role="user")

    if reflection_piece is not None:
        reflecting_message = (
            "You've previously attempted this. Try to improve the performance based on "
            f"the following reflection. {reflection_piece}"
        )
        logger.info("Injecting reflection message | chars=%s", len(reflecting_message))
        agent.update(content=reflecting_message, role="user")

    got_result = False
    num_iter = 0
    max_iter = 10
    failed = 0

    while (not got_result) and num_iter <= max_iter:
        num_iter += 1
        logger.info("API solve iteration %s/%s", num_iter, max_iter)

        reply = agent.step(stop=agent.stop)
        logger.info("Assistant reply received | chars=%s", len(reply))

        returned_code = extract_code(reply)
        logger.info("Extracted code from reply | code_chars=%s", len(returned_code))

        violation = add_gaurdrail_for_no_api(args, returned_code)
        if violation is not None:
            logger.info("Guardrail violation detected: non-permitted API usage.")
            agent.update(role="user", content=violation)
            continue

        new_text = re.sub("\n", "", returned_code, flags=re.IGNORECASE)
        if len(new_text) == 0:
            logger.info("No code detected in current iteration.")
            agent.update(role="user", content=no_code_feedback)
            output = ""
        else:
            logger.info("Building executable payload from generated code.")
            code_to_execute = add_execution_string(args, returned_code)
            logger.info("Executing generated code | payload_chars=%s", len(code_to_execute))
            output = redirect_stdout(code_to_execute, global_dict, local_dict)
            logger.info("Generated code execution complete | output_chars=%s", len(output))

            program_output = iteration_program_output(output)
            agent.update(role="user", content=program_output)

        if "An error occurred:" in output:
            if failed >= 5:
                logger.warning("Too many execution errors (%s). Returning NaN result.", failed)
                return reply, agent.chat, user_message, "The result is: nan"
            failed += 1
            logger.info("Execution error detected. failure_count=%s", failed)
            continue

        if (
            "[SUCCESS]" in reply
            or "SUCCESS" in reply
            or "SUCCEESS" in reply
            or num_iter == max_iter
            or ("Llama-3-70b" in args.openai and "def solver(" in reply)
        ):
            got_result = True
            logger.info("Result condition met at iteration=%s. Running evaluation.", num_iter)
            m_mse = evaluating_output(args, write_result=write_result)
            agent.save_chat(result=m_mse, trial=trial)
            logger.info("Agent_with_API completed | trial=%s metric=%s", trial, m_mse)
            return reply, agent.chat, user_message, m_mse

    logger.warning("Agent_with_API exited loop without success marker.")
    return reply, agent.chat, user_message, "The result is: nan"


def Agent_based_on_text(
    openai_key: str,
    system_prompt: str,
    global_dict,
    local_dict,
    model="gpt-3.5-turbo-0613",
    temperature=0.2,
    top_p=0.1,
    args=None,
    write_result=True,
):
    logger.info("Agent_based_on_text start | model=%s temperature=%s top_p=%s", model, temperature, top_p)
    openai.api_key = openai_key
    if "Llama" in args.openai:
        openai.api_base = args.base_url
        logger.info("Configured Together-compatible OpenAI base_url=%s", args.base_url)

    agent = OpenAIAgent(args, model=model, system_prompt=system_prompt[0], temperature=temperature, top_p=top_p)

    context_len = len(system_prompt[1])
    logger.info("Text prompt context length=%s", context_len)
    if "gpt-3.5" in args.openai and context_len > 12000:
        mse = np.nan
        logger.warning("Context exceeds gpt-3.5 limit. Returning NaN.")
        if args.write_to_csv:
            write_to_csv_file(args.mode, args.query, args.index, args.log_name, mse)
        return
    if "Llama" in args.openai and context_len > 6000:
        mse = np.nan
        logger.warning("Context exceeds Llama limit. Returning NaN.")
        if args.write_to_csv:
            write_to_csv_file(args.mode, args.query, args.index, args.log_name, mse)
        return
    if "gpt-4" in args.openai and context_len > 217600:
        mse = np.nan
        logger.warning("Context exceeds gpt-4 limit. Returning NaN.")
        if args.write_to_csv:
            write_to_csv_file(args.mode, args.query, args.index, args.log_name, mse)
        return

    user_message = format_user_query(args)
    agent.update(role="user", content=system_prompt[1] + " " + user_message)

    got_result = False
    iter_num = 0
    while not got_result and iter_num <= 5:
        iter_num += 1
        logger.info("Text-mode iteration %s/5", iter_num)

        reply = agent.step()
        logger.info("Text-mode assistant reply chars=%s", len(reply))
        logger.info("Text-mode result obtained; parsing numeric array.")

        input_array = extract_array_from_str(reply)
        logger.info("Extracted output array length=%s", len(input_array))

        m_mse = evaluating_output(args, input_array=input_array, write_result=write_result)
        agent.save_chat(result=m_mse)
        logger.info("Agent_based_on_text completed | metric=%s", m_mse)
        return reply


def safe_execution_once(args, reply, global_dict, local_dict, verbose=True):
    returned_code = extract_code(reply)
    logger.info("safe_execution_once start | verbose=%s code_chars=%s", verbose, len(returned_code))

    new_text = re.sub("\n", "", returned_code, flags=re.IGNORECASE)
    if len(new_text) == 0:
        output = "No code is detected in the current round!"
        logger.info("safe_execution_once found no executable code.")
        return output, reply

    code_to_execute = add_execution_string(args, returned_code)
    output = redirect_stdout(code_to_execute, global_dict, local_dict)
    logger.info("safe_execution_once execution completed | output_chars=%s", len(output))

    if verbose:
        program_output = iteration_program_output(output)
    else:
        program_output = output
    return program_output, returned_code
