import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from models.gpt3_5 import GPT3_5
from models.gpt4 import GPT4
from models.codellama_vllm_api import CodeLlamaVLLMAPI
from human_eval.execution import check_correctness
from human_eval.data import read_problems
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AtomicLogger import AtomicLogger, LogLevel


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
        choices=["codellama", "chatgpt", "gpt4"],
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
        '--num-threads',
        required=False,
        type=int,
        default=2,
        help='The number of threads to use for parallelizing *on the task level*.',
    )
    return parser


def main(args):

    logger = AtomicLogger(None, level=LogLevel.DEBUG)

    problems = read_problems()["test"]

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
                p = j["task_id"]
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
            problems = {p: problems[p] for p in problems if p not in completed_problems}

    # Load the model
    if args.model == "chatgpt":
        model = GPT3_5()
    elif args.model == "gpt4":
        model = GPT4()
    elif args.model == "codellama":
        model = CodeLlamaVLLMAPI()
    else:
        raise ValueError(f"Unknown model {args.model}")

    def do(problem):
        id = problem["task_id"]
        logger = AtomicLogger(None, level=LogLevel.DEBUG)
        logger.add_log_flush(
            "main", f"Getting completions for {id}", LogLevel.INFO
        )

        context = model.get_generation_context()
        prompt = problem['prompt']
        prompt = model.open_delim + '\n' + prompt + '\n' + model.close_delim
        prompt = model.construct_prompt(context, prompt)

        logger.add_log_flush("main", f"Prompt: {prompt}", LogLevel.DEBUG)

        completions = model.get_completions(
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
                f"Got {len(completions)} completions for problem {id} but expected {args.num_samples}",
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
                    p.submit(check_correctness, problem, code, 3.): (code, completion)
                    for code, completion in zip(codes, completions)
                }
                for future in as_completed(future_to_completion):
                    result = future.result()
                    code, completion = future_to_completion[future]
                    results.append((result["passed"], result["result"], code, completion))
            except Exception as e:
                logger.add_log_flush(
                    "main",
                    f"Error during execution of completions for problem {id}: {e}",
                    LogLevel.ERROR,
                )
                raise e

        logger.add_log_flush(
            "main",
            f"Writing results for problem {id} to {args.output}.",
            LogLevel.INFO,
        )

        with open(args.output, "a") as f:
            f.write(
                json.dumps(
                    {
                        "task_id": id,
                        "prompt": prompt,
                        "completions": [
                            {
                                "original_completion": original_completion,
                                "executed_completion": code,
                                "binary": binary_result,
                                "fault": fault,
                            }
                            for binary_result, fault, code, original_completion in results
                        ],
                    }
                )
                + "\n"
            )

    logger.add_log_flush("main", f"Starting. Getting completions for {len(problems)} problems.", LogLevel.INFO)
    bar = tqdm(total=len(problems), desc='Problems handled')
    sorted_problem_keys = sorted(problems.keys())
    if args.num_threads == 1:
        for k in sorted_problem_keys:
            do(problems[k])
            bar.update(n=1)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as p:
            fs = [p.submit(do, problems[k]) for k in sorted_problem_keys]
            for _ in as_completed(fs):
                bar.update(n=1)

    logger.add_log_flush("main", f"Done. Wrote outputs to {args.output}", LogLevel.INFO)
    logger.flush()  # this SHOULD be a no-op


if __name__ == "__main__":
    arg_parser = build_args()
    args = arg_parser.parse_args()
    main(args)
