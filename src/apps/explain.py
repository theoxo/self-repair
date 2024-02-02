import os
import sys
import json
import argparse
import openai
from tqdm import tqdm
openai.api_key = os.getenv("OPENAI_API_KEY")
# add .. to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AtomicLogger import AtomicLogger, LogLevel
from apps import (
    get_error_message,
    get_difficulty,
)
from models.gpt4 import GPT4
from models.gpt3_5 import GPT3_5
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

APPS_DIR = os.environ["APPS_DIR"]


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, type=str, help="The file containing the samples"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Final results will be written to this dataset.",
    )
    parser.add_argument(
        "--logdir",
        required=False,
        default=".",
        type=str,
        help="The dir to write the log to.",
    )
    parser.add_argument(
        "--logfile",
        required=False,
        default=None,
        type=str,
        help="The file to write the log to. Default uses datetime.",
    )
    parser.add_argument("--n", required=False, default=25, type=int)
    parser.add_argument(
        "--temperature",
        required=False,
        default=0.8,
        type=float,
        help="The temperature to use for the completion model",
    )
    parser.add_argument("--max-tokens", required=False, default=512, type=int)
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="The model to use for explanation completions",
        choices=["gpt4", "chatgpt"],
    )
    parser.add_argument(
        '--num-threads',
        required=False,
        type=int,
        default=2,
        help='The number of threads to use for parallelizing *on the task level*.',
    )
    return parser


def clean_explanation(explanation):
    exp = explanation
    if '###' in exp:
        exp = exp[:exp.find('###')]
    if "```python" in exp:
        exp = exp[: exp.find("```python")]
    exp = exp.strip()
    return exp

def explain_code(
    prob_path,
    spec,
    code,
    errors,
    model,
    n=25,
    temperature=0.8,
    max_tokens=256,
    parent_logger=None,
):

    if n == 0:
        return []

    args = locals()
    args.pop("parent_logger")
    args["model"] = model.get_model_name()

    logger = (
        None
        if parent_logger is None
        else AtomicLogger(
            parent_logger.logfile, parent_logger.prints, parent_logger.log_level
        )
    )

    error_msg = get_error_message(prob_path, errors, logger=logger)

    if logger is not None:
        logger.add_log(
            "explain_code",
            "starting with args: " + json.dumps(args),
            log_level=LogLevel.DEBUG,
        )

    explanations, _ = model.generate_explanation_completions(
        spec, code, error_msg, temperature=temperature, max_tokens=max_tokens, logger=logger, num_samples=n
    )

    explanations = [clean_explanation(exp) for exp in explanations]

    if logger is not None:
        logger.add_log(
            "explain_code",
            "completed with args: " + json.dumps(args),
            log_level=LogLevel.DEBUG,
        )
        logger.flush()


    return explanations


def get_model(s):
    # Load the model
    if s not in ["gpt4", "chatgpt"]:
        raise ValueError(f'Unknown or unsupported(?) model {s}')
    if s == "gpt4":
        return GPT4()
    elif s == "chatgpt":
        return GPT3_5()


def main(args):
    if args.logfile:
        logfile = args.logfile
    else:
        import datetime

        current_time = datetime.datetime.now()
        logfile = (
            "explain_log"
            f"_dt={current_time.year}"
            f"-{current_time.month}"
            f"-{current_time.day}"
            f"-{current_time.hour}"
            f"-{current_time.minute}"
            f"-{current_time.second}"
            ".log"
        )
    logfile = os.path.join(args.logdir, logfile)

    logger = AtomicLogger(logfile, level=LogLevel.INFO)

    logger.add_log_flush(
        "main", f"running with args: {json.dumps(vars(args))}", log_level=LogLevel.INFO
    )

    ps = {}
    with open(args.input, "r") as f:
        for line in f.readlines():
            line = line.strip()
            j = json.loads(line)
            p = j["prob_path"]
            if APPS_DIR not in p:
                p = p.replace("/home/theo/ai4code/APPS", APPS_DIR)
            if APPS_DIR not in p:
                p = p.replace("/scratch/theoxo/apps/APPS", APPS_DIR)
            assert p not in ps, (p, ps[p], j)
            ps[p] = j

    length_before = len(ps)

    if args.output and os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f.readlines():
                line = line.strip()
                j = json.loads(line)
                if len(j.keys()) == 1:
                    p = list(j.keys())[0]
                else:
                    p = j["prob_path"]
                if APPS_DIR not in p:
                    p = p.replace("/home/theo/ai4code/APPS", APPS_DIR)
                if APPS_DIR not in p:
                    p = p.replace("/scratch/theoxo/apps/APPS", APPS_DIR)
                logger.add_log_flush(
                    "main",
                    f"Skipping problem {p} because we already have output for it",
                    log_level=LogLevel.INFO,
                )
                assert p in ps, f"problem {p} not in input file but in output file"
                ps.pop(p)

    length_after = len(ps)
    logger.add_log_flush(
        "main",
        f"filtered out {length_before - length_after} problems that were already in the output file",
        log_level=LogLevel.INFO,
    )

    model = get_model(args.model)

    def do(problem, problem_json):
        jsons = problem_json["completions"]
        local_logger = AtomicLogger(logger.logfile, level=logger.log_level)
        local_logger.add_log(
            "main",
            f"trying to explain {len(jsons)} entries for prob_path {problem}",
            log_level=LogLevel.INFO,
        )

        for j in jsons:
            if isinstance(j["binary"], str):
                j["binary"] = j["binary"] == "True"  # for legacy reasons

        with open(f"{problem}/question.txt", "r") as f:
            lines = f.readlines()
            lines = [f"# {line.strip()}" for line in lines]
            spec = "\n".join(lines)

        results = [j for j in jsons if j["binary"]]
        if any(not j["binary"] for j in jsons):
            local_logger.add_log(
                "main",
                f"explaining {len(jsons) - len(results)} entries for prob_path {problem}",
                log_level=LogLevel.INFO,
            )
            bar = tqdm(total=len([j for j in jsons if not j["binary"]]), desc="Base programs explained")
            with ProcessPoolExecutor(max_workers=1) as p:
                future_to_json = {
                    p.submit(
                        explain_code,
                        problem,
                        spec,
                        j["executed_completion"],
                        j["errors"],
                        model,
                        n=args.n - len([e for e in j.get('explanations', []) if e[-1] == '.']),
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        parent_logger=local_logger,
                    ): j
                    for j in jsons
                    if not j["binary"]
                }
                for future in as_completed(future_to_json):
                    bar.update(n=1)
                    j = future_to_json[future]
                    explanations = [e for e in j.get("explanations", []) if e[-1] == '.']
                    explanations.extend(future.result())
                    j["explanations"] = explanations
                    results.append(j)
        else:
            local_logger.add_log(
                "main",
                f"no need to process {len(jsons)} entries for prob_path {problem}",
                log_level=LogLevel.INFO,
            )

        problem_json["completions"] = results
        local_logger.add_log_flush(
            "main",
            f"finished processing {len(jsons)} entries for prob_path {problem}",
            log_level=LogLevel.INFO,
        )
        if args.output:
            with open(args.output, "a") as f:
                f.write(json.dumps(problem_json) + "\n")


    # sort the problems by difficulty so that we can do the easy ones first
    difficulties = ['introductory', 'interview', 'competition']
    ps = sorted(ps.items(), key=lambda x: x[0])  # sort by index
    ps = sorted(ps, key=lambda x: difficulties.index(get_difficulty(x[0])))  # sort by difficulty

    bar = tqdm(total=len(ps), desc="Problems handled")
    if args.num_threads == 1:
        for p, jsons in ps:
            do(p, jsons)
            bar.update(n=1)

    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as p:
            fs = [
                p.submit(do, p, problem_json)
                for (p, problem_json) in ps
            ]
            for _ in as_completed(fs):
                bar.update(n=1)

    logger.add_log_flush(
        "main",
        f"Finished experiment. See log file {logfile} for details.",
        log_level=LogLevel.INFO,
    )
    if args.output:
        logger.add_log_flush(
            "main",
            f"Produced dataset of programs written to {args.output}.",
            log_level=LogLevel.INFO,
        )


if __name__ == "__main__":
    main(build_args().parse_args())
