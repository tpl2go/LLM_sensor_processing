import logging
import os
from datetime import datetime

from utils import openai_api, extract_code, redirect_stdout

logger = logging.getLogger(__name__)


class OpenAIAgent:
    def __init__(self, args, model, system_prompt, temperature, top_p):
        self.args = args
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.model = model
        self.chat = [{"role": "system", "content": self.system_prompt}]

        now = datetime.now()
        current_time_str = now.strftime("%m-%d %H:%M:%S")

        name = f"{args.query}_{args.index}_{args.mode}_{args.eval}_{args.num_trial}_{current_time_str}"
        conv_dir = "./conv_history/{}/".format(model)
        if not os.path.isdir(conv_dir):
            os.makedirs(conv_dir, exist_ok=True)
        self.file_name = conv_dir + name

        if args.mode == "base":
            self.stop = None
        else:
            self.stop = ["```\n", "```\n\n", "</s>"]

        logger.info(
            "OpenAIAgent initialized | model=%s mode=%s chat_len=%s history_file=%s",
            self.model,
            self.args.mode,
            len(self.chat),
            self.file_name,
        )

    def update(self, content, role):
        self.chat.append({"role": role, "content": content})
        logger.info("Chat updated | role=%s total_messages=%s", role, len(self.chat))

    def step(self, stop=None):
        logger.info(
            "Requesting model step | model=%s messages=%s temperature=%s top_p=%s stop=%s",
            self.model,
            len(self.chat),
            self.temperature,
            self.top_p,
            stop,
        )
        message = openai_api(
            self.chat,
            self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop,
            request_timeout=getattr(self.args, "api_timeout", 180.0),
        )
        if "```Python" in message or "```python" in message:
            message += "```\n"
        self.update(message, "assistant")
        logger.info("Model step received | response_chars=%s", len(message))
        return message

    def reset(self):
        # only the system prompt is kept
        self.chat = self.chat[0]

    def save_chat(self, trial=1, result="None"):
        out_path = self.file_name + f"_trial_{trial}.txt"
        with open(out_path, "w", encoding="utf-8") as file:
            for item in self.chat:
                file.write(str(item) + "\n")
            if self.args.target_file is not None:
                file.write(result)
        logger.info("Saved chat history to %s", out_path)


class ReflectOpenAIAgent(OpenAIAgent):
    def __init__(self, args, model, system_prompt, temperature, top_p):
        super(OpenAIAgent).__init__()

        self.args = args
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.model = model
        self.chat = [{"role": "system", "content": self.system_prompt}]
        self.performance_hist = []

        now = datetime.now()
        current_time_str = now.strftime("%m-%d %H:%M:%S")

        name = f"Reflector_{args.query}_{args.index}_{args.mode}_{args.eval}_{args.num_trial}_{current_time_str}"
        conv_dir = "./conv_history/{}/".format(model)
        if not os.path.isdir(conv_dir):
            os.makedirs(conv_dir, exist_ok=True)
        self.file_name = conv_dir + name
        logger.info("ReflectOpenAIAgent initialized | history_file=%s", self.file_name)

    def reset(self):
        self.chat = [{"role": "system", "content": self.system_prompt}]
        logger.info("ReflectOpenAIAgent state reset.")

    def update(self, context, question, performance):
        self.performance_hist.append(performance)

        perf_hist = [
            f"In trial #{i+1}, the performance is - " + self.performance_hist[i]
            for i in range(len(self.performance_hist))
        ]
        perf_hist = "An external source perform evalution on your output signal w.r.t. the ground truth signal. " + ". ".join(perf_hist)
        self.chat[0]["content"] = self.chat[0]["content"].format(
            context=context[2:], question=question, performance=performance, performance_hist=perf_hist
        )
        logger.info("ReflectOpenAIAgent updated | performance_history_size=%s", len(self.performance_hist))

    def step(self, trial=0):
        logger.info("ReflectOpenAIAgent step start | trial=%s", trial)
        message = openai_api(
            self.chat,
            self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            request_timeout=getattr(self.args, "api_timeout", 180.0),
        )
        logger.info("ReflectOpenAIAgent response received | chars=%s", len(message))
        self.chat.append({"role": "assistant", "content": message})
        self.save_chat(trial=trial)
        return message


class EvalOpenAIAgent(OpenAIAgent):
    def __init__(self, args, model, system_prompt, temperature, top_p):
        super(OpenAIAgent).__init__()

        self.args = args
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.model = model
        self.chat = [{"role": "system", "content": self.system_prompt}]
        self.performance_hist = []
        self.memory = []
        self.memory_str = ""

        now = datetime.now()
        current_time_str = now.strftime("%m-%d %H:%M:%S")

        name = f"{args.eval}_{args.query}_{args.index}_{args.mode}_{args.eval}_{args.num_trial}_{current_time_str}"
        conv_dir = "./conv_history/{}/".format(model)
        if not os.path.isdir(conv_dir):
            os.makedirs(conv_dir, exist_ok=True)
        self.file_name = conv_dir + name
        logger.info("EvalOpenAIAgent initialized | history_file=%s", self.file_name)

    def extract_result(self, result: str):
        start = result.find("EVALUATION")
        end = -1
        return result[start:end]

    def update_memory(self, result):
        self.memory.append(result)
        self.memory_str = [f"In trial #{i+1}, your evaluation is - " + self.memory[i] for i in range(len(self.memory))]
        self.memory_str = " ".join(self.memory_str)
        logger.info("Eval memory updated | size=%s", len(self.memory))

    def reset(self):
        self.chat = [{"role": "system", "content": self.system_prompt}]
        logger.info("EvalOpenAIAgent state reset.")

    def init(self, context, question, vis_result=None):
        self.chat[0]["content"] = self.chat[0]["content"].format(
            context=context[2:], question=question, memory=self.memory_str, vis_result=vis_result
        )
        logger.info("EvalOpenAIAgent prompt initialized | context_messages=%s", len(context))

    def update(self, content, role):
        self.chat.append({"role": role, "content": content})
        logger.info("Eval chat updated | role=%s total_messages=%s", role, len(self.chat))

    def step(self, stop=None):
        logger.info("EvalOpenAIAgent step start | stop=%s", stop)
        message = openai_api(
            self.chat,
            self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop,
            request_timeout=getattr(self.args, "api_timeout", 180.0),
        )

        if "```Python" in message or "```python" in message:
            message += "```\n"

        logger.info("EvalOpenAIAgent response received | chars=%s", len(message))
        self.update(message, "assistant")
        return message

    def eval(self, context, question, global_dict, local_dict, trial=0):
        logger.info("Evaluator start | trial=%s", trial)
        vis_output_str = """
from utils import read_data, store_data
output_data, sampling_rate = read_data(args.output_file)
input_data, sampling_rate = read_data(args.input_file)
print(f"The produced output_data is: ", output_data)
"""
        vis_result = redirect_stdout(vis_output_str, global_dict, local_dict)
        logger.info("Evaluator visualization context prepared | chars=%s", len(vis_result))

        self.init(context, question, vis_result)
        reply = ""
        i = 0
        succeeded = False
        failed = 0
        while i < 5 and not succeeded:
            logger.info("Evaluator iteration %s/5", i + 1)
            reply = self.step(stop=["```\n", "```\n\n", "</s>"])
            code = extract_code(reply)
            result = ""
            logger.info("Evaluator extracted code length=%s", len(code))
            if len(code) == 0:
                if "[EVALUATION]" in reply:
                    succeeded = True
                self.update(
                    content="Please go ahead. Remember to put your final evaluation after [EVALUATION] and the iteration will stop.",
                    role="user",
                )
            else:
                code_to_execute = "\n" + code
                code_to_execute += """
from utils import read_data, store_data
output_data, sampling_rate = read_data(args.output_file)
input_data, sampling_rate = read_data(args.input_file)
"""

                if "def inspection(" in code_to_execute:
                    code_to_execute += "inspect_result = inspection(input_data, output_data, sampling_rate)\n"
                    code_to_execute += "challenge_feedback(inspect_result, inspection=True)\n"
                elif "def challenger(" in code_to_execute or "def verifier(" in code_to_execute:
                    if self.args.eval == "self_coding":
                        code_to_execute += "result = challenger(input_data, output_data, sampling_rate)\n"
                    else:
                        code_to_execute += "result = verifier()\n"
                    code_to_execute += "challenge_feedback(result)\n"

                result = redirect_stdout(code_to_execute, global_dict, local_dict)
                logger.info("Evaluator code execution complete | output_chars=%s", len(result))

                if len(result) == 0:
                    result += "The above program prints nothing. If it is not intended, remember to use print() function. Remember to put your final evaluation after [EVALUATION] and the iteration will stop."
                else:
                    result = "The program output: " + result

                self.update(content=result, role="user")

                if "An error occurred:" in result:
                    if failed >= 3:
                        logger.warning("Evaluator exceeded max execution failures.")
                        return "The challenge/verification result is: False"
                    failed += 1
                    logger.info("Evaluator execution failure count=%s", failed)
                    continue

                if "[EVALUATION]" in reply or "The challenge/verification result is: " in result:
                    succeeded = True
                    logger.info("Evaluator success condition met.")
            i += 1

        evaluation = reply + "\n" + result
        self.update_memory(evaluation)
        self.save_chat(trial=trial)
        self.reset()
        logger.info("Evaluator finished | trial=%s", trial)
        return evaluation
