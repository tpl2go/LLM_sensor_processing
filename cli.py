import argparse
import logging
import os

from constr_system_prompt import SystemPrompt, SystemPromptECGPPG
from chat import Agent_based_on_text, Agent_with_reflection

global_dict, local_dict = globals(), locals()
logger = logging.getLogger(__name__)


def configure_logging(log_level: str = "INFO", log_file: str = None):
    numeric_level = getattr(logging, str(log_level).upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logger.info("Logging initialized at level=%s log_file=%s", log_level, log_file)


def _resolve_input_files(args):
    if args.query is None or args.index is None:
        raise ValueError("--query and --index are required for benchmark execution.")

    if "speech" in args.query:
        return [f"./benchmark/{args.query}/{args.index}.wav"]
    if "imputation" in args.query or "extrapolation" in args.query:
        return [f"./benchmark/{args.query}/{args.index}_50.npy"]
    if "gait-delay_detection" in args.query:
        return [
            f"./benchmark/{args.query}/{args.index}_1.npy",
            f"./benchmark/{args.query}/{args.index}_2.npy",
        ]
    if "gait-period_detection" in args.query:
        return [f"./benchmark/{args.query}/{args.index}_1.npy"]
    return [f"./benchmark/{args.query}/{args.index}.npy"]


def main(args):
    logger.info(
        "Run start | mode=%s query=%s model=%s index=%s num_trial=%s eval=%s",
        args.mode,
        args.query,
        args.openai,
        args.index,
        args.num_trial,
        args.eval,
    )

    args.input_file = _resolve_input_files(args)
    logger.info("Resolved input_file=%s", args.input_file)

    valid_models = (
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4o",
        "gpt-4-0125-preview",
        "gpt-4-turbo",
        "gpt-4o-mini",
        "o1",
        "o3-mini",
        "Llama-2-70b",
        "Llama-2-13b",
        "Llama-2-7b",
        "Llama-3-8b",
        "Llama-3-70b",
        "Qwen1.5-110B",
        "Qwen2-72B",
    )
    if args.openai not in valid_models:
        raise ValueError(f"Unsupported model: {args.openai}")

    prefix = ""
    original_model = args.openai
    if "Llama" in args.openai:
        prefix = "meta-llama/"
        args.openai = prefix + args.openai + "-chat-hf"
    elif "Qwen1.5" in args.openai:
        prefix = "Qwen/"
        args.openai = prefix + args.openai + "-Chat"
    elif "Qwen2" in args.openai:
        prefix = "Qwen/"
        args.openai = prefix + args.openai + "-Instruct"
    elif "Mixtral" in args.openai:
        prefix = "mistralai/"
        args.openai = prefix + args.openai + "-Instruct-v0.1"
    logger.info("Model normalization | input=%s resolved=%s", original_model, args.openai)

    if "ecg" in args.input_file[0]:
        args.file = "ecg_data"
    elif "ppg" in args.input_file[0]:
        args.file = "ppg"
    else:
        args.file = "general"
    logger.info("Detected file family: %s", args.file)

    target_file = args.input_file[0].split("/")
    filename_parts = target_file[-1].split(".")
    filename_parts[0] = filename_parts[0] + "_gt"
    filename = ".".join(filename_parts)
    target_file[-1] = filename
    if "VoiceDetector" in args.input_file[0]:
        target_file[-1] = filename.replace("wav", "npy")
    args.target_file = "/".join(target_file)
    logger.info("Resolved target_file=%s", args.target_file)

    output_dir = "./llm_response/"
    args.output_file = (
        output_dir
        + f"{args.openai}_{args.query}_{args.index}_{args.num_trial}."
        + filename.split(".")[-1]
    )
    os.makedirs(output_dir + prefix, exist_ok=True)

    if "VoiceDetector" in args.input_file[0]:
        args.output_file = output_dir + f"{args.openai}_{args.query}_{args.index}_{args.num_trial}.npy"
    args.log_name = f"{args.openai}_{args.mode}_{args.eval}_#trial_{args.num_trial}"
    logger.info("Resolved output_file=%s", args.output_file)
    logger.info("Resolved log_name=%s", args.log_name)

    if args.adaptive_reflect and (
        "extrapolation" in args.input_file[0] or "imputation" in args.input_file[0]
    ):
        logger.info("adaptive_reflect enabled for extrapolation/imputation. Forcing num_trial=1.")
        args.num_trial = 1

    try:
        if (
            "gpt-3" in args.openai
            or "gpt-4" in args.openai
            or "o1" in args.openai
            or "o3-mini" in args.openai
            or "Llama" in args.openai
            or "Qwen" in args.openai
            or "Mixtral" in args.openai
        ):
            model = args.openai
            logger.info("Selected model for runtime: %s", model)

            using_together = "Llama" in args.openai or "Qwen" in args.openai or "Mixtral" in args.openai
            key_path = "together_key.txt" if using_together else "key.txt"
            logger.info("Loading API key from %s", key_path)
            with open(key_path, encoding="utf-8") as f:
                openai_key = f.read().strip()
            logger.info("API key loaded (non-empty=%s)", bool(openai_key))

            if not using_together:
                os.environ["OPENAI_API_KEY"] = openai_key
                logger.info("Set OPENAI_API_KEY environment variable.")

            args.system_prompt_file = f"./sys/system_prompt_signal_processing_{args.mode}.txt"
            logger.info("Using system prompt file: %s", args.system_prompt_file)
            if not os.path.exists(args.system_prompt_file):
                logger.warning("System prompt file does not exist: %s", args.system_prompt_file)

            if args.mode == "text":
                logger.info("Constructing text-mode system prompt...")
                system_prompt = SystemPromptECGPPG(
                    system_prompt_file=args.system_prompt_file,
                    length=args.ts_len,
                    mode=args.mode,
                    args=args,
                )
                logger.info("System prompt ready. Entering Agent_based_on_text.")
                Agent_based_on_text(
                    openai_key,
                    system_prompt.system_prompt,
                    global_dict,
                    local_dict,
                    model=model,
                    temperature=0.8,
                    top_p=1,
                    args=args,
                )
            elif args.mode in ("api", "no_api", "CoT", "react", "base"):
                logger.info("Constructing API-mode system prompt...")
                system_prompt = SystemPrompt(
                    imu_file=args.imu_file,
                    geo_file=args.geo_file,
                    input_file=args.input_file,
                    system_prompt_file=args.system_prompt_file,
                    args=args,
                )
                logger.info("System prompt ready. Entering Agent_with_reflection.")
                Agent_with_reflection(
                    openai_key,
                    system_prompt.system_prompt,
                    global_dict,
                    local_dict,
                    model=model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    args=args,
                )
            else:
                logger.warning("Unsupported mode '%s' for execution path.", args.mode)
        else:
            logger.warning("No matching runtime path found for model=%s", args.openai)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user (KeyboardInterrupt).")
    except Exception:
        logger.exception("Unhandled exception during run.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai", type=str, default="gpt-4o", help="default use gpt-4 model")
    parser.add_argument("--imu_file", type=str, default="./data/sample.csv", help="default imu data")
    parser.add_argument("--geo_file", type=str, default="./data/geo.csv", help="default geolocation data")
    parser.add_argument("--input_file", type=str, default=None, nargs="+", help="default heart rate data")
    parser.add_argument("--target_file", type=str, default=None, help="default heart rate data")
    parser.add_argument("--output_file", type=str, default=None, help="The file that you want the model to produce.")
    parser.add_argument("--system_prompt_file", type=str, default="./system_prompt.txt", help="default system prompt")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--mode", type=str, default="code", help="Conversational AI mode")
    parser.add_argument("--index", type=str, default=None, help="file index")
    parser.add_argument("--file", type=str, default=None, help="-")
    parser.add_argument("--ts_len", type=int, default=None, help="Time series sequence length")
    parser.add_argument("--CoT", action="store_true", help="Use chain of thought")
    parser.add_argument("--knowledge_signal", action="store_true", help="Whether to inject signal knowledge")
    parser.add_argument(
        "--adaptive_reflect",
        action="store_true",
        help=(
            "Adaptively reflect on its solution. When it is True, "
            "number of reflection will be set to 1 for imputation and extrapolation."
        ),
    )
    parser.add_argument("--knowledge_task", action="store_true")
    parser.add_argument("--write_to_csv", action="store_true", help="write results to csv file")
    parser.add_argument("--query", type=str, default=None, help="user's query for testing")
    parser.add_argument(
        "--encode",
        type=str,
        default="env",
        help="Ways to present sequences to the modes. They include: number, space, and alpabet.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="self_coding",
        help="Feedback from the environment or self-generated. (env | self_vis | self_coding | self_verifier)",
    )
    parser.add_argument(
        "--bw_pred",
        type=int,
        default=0,
        help="Whether we want the model to do backward extrapolation (if bw_pred >= 1)",
    )
    parser.add_argument("--num_trial", type=int, default=1, help="How many times can the model reflect and retry")
    parser.add_argument("--base_url", type=str, default="https://api.together.xyz/v1", help="together.ai interface")
    parser.add_argument("--log_name", type=str, default="test", help="The type of task we are testing.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Python logging level (DEBUG, INFO, WARNING)")
    parser.add_argument("--log_file", type=str, default=None, help="Optional path to write logs.")
    parser.add_argument(
        "--api_timeout",
        type=float,
        default=180.0,
        help="Timeout (seconds) for each OpenAI/Together API request attempt.",
    )
    args = parser.parse_args()
    configure_logging(args.log_level, args.log_file)
    main(args)
