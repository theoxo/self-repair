import torch
import tiktoken
import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from models.chatgpt import ChatGPT
from models.gpt4 import GPT4
from models.codellama_vllm import CodeLlamaVLLM
from apps import exec_sample, get_starter_code, comment, is_call_based, get_tests
from colorama import Fore
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AtomicLogger import AtomicLogger, LogLevel

APPS_DIR = os.environ["APPS_DIR"]



def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", required=True, type=str, help="The file to write the output to"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="The model to use",
        choices=["codellama_vllm", "chatgpt", "gpt4"],
    )
    parser.add_argument(
        "--num-samples", default=50, type=int, help="The number of samples to take"
    )
    parser.add_argument(
        "--skip-if-at",
        default=None,
        type=int,
        help="If the output file already contains this many samples, skip the problem",
    )
    parser.add_argument(
        "--temperature",
        default=0.8,
        type=float,
        help="The temperature to use for sampling",
    )
    parser.add_argument(
        "--max-tokens",
        default=512,
        type=int,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument(
        '--problems-json',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--num-threads',
        required=False,
        type=int,
        default=2,
        help='The number of threads to use for parallelizing *on the task level*.',
    )
    parser.add_argument(
        '--split',
        required=False,
        type=int,
        default=None,
    )
    parser.add_argument(
        '--out-of',
        required=False,
        type=int,
        default=2,
    )
    parser.add_argument(
        '--seed',
        required=False,
        type=int,
        default=538,
    )
    return parser


def main(args):

    gpt_enc = tiktoken.get_encoding("cl100k_base")

    torch.manual_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    logger = AtomicLogger(None, level=LogLevel.DEBUG if local_rank < 1 else LogLevel.ERROR)

    with open(args.problems_json, "r") as f:
        problems = json.load(f)

    if (
        args.skip_if_at is not None
        and args.output is not None
        and os.path.exists(args.output)
    ):
        completed_problems = set()
        scanned_problems = {}
        with open(args.output, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                try:
                    j = json.loads(line)
                except Exception as e:
                    logger.add_log_flush(
                        "main", f"Error parsing line #{idx}: {line}", LogLevel.ERROR
                    )
                    raise e
                p = j["prob_path"].split('/')[-1]
                if p not in scanned_problems:
                    scanned_problems[p] = 0
                scanned_problems[p] += len(j["completions"])

                if scanned_problems[p] >= args.skip_if_at:
                    completed_problems.add(p)

        if len(completed_problems) > 0:
            logger.add_log_flush(
                "main",
                f"Skipping {len(completed_problems)} problems that have already been completed: {completed_problems}",
                LogLevel.INFO,
            )
            problems = list(filter(lambda p: p not in completed_problems, problems))

    problems = sorted(problems)

    if args.split is not None:
        len_before = len(problems)
        problems = [p for i, p in enumerate(problems) if i % args.out_of == args.split]
        logger.add_log_flush(
            "main",
            f"Splitting {len_before} problems into {args.out_of} parts, keeping part {args.split} ({len(problems)} problems)",
            LogLevel.INFO,
        )

    # Load the model
    if args.model == "chatgpt":
        model = ChatGPT()
    elif args.model == "gpt4":
        model = GPT4()
    elif args.model == "codellama_vllm":
        model = CodeLlamaVLLM(args.seed)
    else:
        raise ValueError(f"Unknown model {args.model}")

    def do(problem):
        logger = AtomicLogger(None, level=LogLevel.INFO)
        problem_str = f"{problem}".zfill(4)
        logger.add_log_flush(
            "main", f"Getting completions for {problem_str}", LogLevel.INFO
        )
        prob_path = os.path.join(APPS_DIR, "test", problem_str)

        if not os.path.isfile(os.path.join(prob_path, "input_output.json")):
            logger.add_log_flush(
                "main", f"No I/O file for problem {problem_str}.", LogLevel.ERROR
            )
            raise FileNotFoundError

        with open(f"{prob_path}/question.txt", "r") as f:
            question = f.read()

        # Construct the prompt
        call_based = is_call_based(prob_path, logger)
        #context = model.get_generation_context(call_based, args.tests_in_prompt)
        context = model.get_generation_context(call_based)
        prompt = model.QUESTION_DELIMITER + "\n" + question + "\n"
        starter_code = get_starter_code(prob_path, logger)
        if starter_code is not None:
            prompt += "### STARTER CODE\n" + comment(starter_code) + "\n"
        if is_call_based(prob_path, logger):
            prompt += "### Use Call-Based Format\n"
        else:
            prompt += "### Use Standard Input Format (read inputs with `input()`, write results with `print()`)\n"

        prompt = model.construct_prompt(context, prompt)

        logger.add_log_flush("main", f"Prompt: {prompt}", LogLevel.DEBUG)

        completions, _ = model.get_completions(
            prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            num_samples=args.num_samples,
            logger=logger,
        )
        codes = [model.extract_code_from_completion(c) for c in completions]
        logger.flush()

        if len(completions) != args.num_samples:
            logger.add_log_flush(
                "main",
                f"Got {len(completions)} completions for problem {problem_str} but expected {args.num_samples}",
                LogLevel.ERROR,
            )
            raise ValueError

        results = []
        logger.add_log_flush(
            "main", f"Executing {len(completions)} completions...", LogLevel.INFO
        )
        with ProcessPoolExecutor() as p:
            try:
                future_to_completion = {
                    p.submit(exec_sample, prob_path, code): (code, completion)
                    for code, completion in zip(codes, completions)
                }
                for future in as_completed(future_to_completion):
                    errors, code = future.result()
                    code, completion = future_to_completion[future]
                    if model.MODEL_NAME.startswith('CodeLlama'):
                        tokens_generated = len(model.generator.tokenizer.encode(code, False, False))
                    elif 'CodeLlama' in model.MODEL_NAME:
                        # HF-style tokenizer
                        tokens_generated = len(model.generator.get_tokenizer().encode(code))
                    else:
                        tokens_generated = len(gpt_enc.encode(code))
                    binary_result = all(
                        isinstance(s, str) and s == "passed" for s in errors
                    )
                    if binary_result:
                        fault = "passed"
                    else:
                        for err in errors:
                            if isinstance(err, str):
                                assert (
                                    err == "passed"
                                ), f'error "{err}" was string but not "passed"'
                            else:
                                assert isinstance(err, tuple)
                                err_type = err[0]
                                if err_type == "No inputs-outputs pairs":
                                    # If there are no i/o pairs, we do NOT want to use this problem
                                    # to evaluate repair; so we will sample a new one
                                    logger.add_log_flush(
                                        "main",
                                        f"No i/o pairs in I/O file for problem {problem_str}",
                                        LogLevel.ERROR,
                                    )
                                    raise ValueError(
                                        f"No i/o pairs in I/O file for problem {problem_str}"
                                    )
                                if (
                                    err_type == "COMPILE ERROR"
                                    or err_type == "RUNTIME ERROR"
                                ):
                                    err_type = err[1][: err[1].find("(")]
                                fault = err_type
                                break
                    results.append((binary_result, fault, errors, code, completion, tokens_generated))
            except Exception as e:
                logger.add_log_flush(
                    "main",
                    f"Error during execution of completions for problem {problem_str}: {e}",
                    LogLevel.ERROR,
                )
                raise e

        logger.add_log_flush(
            "main",
            f"Writing results for problem {problem_str} to {args.output}.",
            LogLevel.INFO,
        )

        # note that we only write results to file if we successfully ran all completions (i.e. no segfaults / environment-crashing errors)
        with open(args.output, "a") as f:
            f.write(
                json.dumps(
                    {
                        "prob_path": prob_path,
                        "prompt": prompt,
                        "avg_tokens_generated_per_completion": sum(r[-1] for r in results) / len(results),
                        "completions": [
                            {
                                "original_completion": original_completion,
                                "executed_completion": code,
                                "binary": binary_result,
                                "fault": fault,
                                "tokens_generated": tokens_generated,
                                "errors": errors,
                            }
                            for binary_result, fault, errors, code, original_completion, tokens_generated in results
                        ],
                    }
                )
                + "\n"
            )

    logger.add_log_flush("main", f"Starting. Getting completions for {len(problems)} problems.", LogLevel.INFO)
    bar = tqdm(total=len(problems), desc='Problems handled')
    if args.num_threads == 1:
        for problem in sorted(problems):
            do(problem)
            bar.update(n=1)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as p:
            fs = [p.submit(do, problem) for problem in sorted(problems)]
            for _ in as_completed(fs):
                bar.update(n=1)

    logger.add_log_flush("main", f"Done. Wrote outputs to {args.output}", LogLevel.INFO)
    logger.flush()  # this SHOULD be a no-op


if __name__ == "__main__":
    arg_parser = build_args()
    args = arg_parser.parse_args()
    main(args)
