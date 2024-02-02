import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import openai
from tqdm import tqdm
from human_eval.execution import check_correctness
from human_eval.data import read_problems
from models.gpt4 import GPT4
from models.gpt3_5 import GPT3_5
from models.codellama_vllm_api import CodeLlamaVLLMAPI

openai.api_key = os.getenv("OPENAI_API_KEY")
from AtomicLogger import AtomicLogger, LogLevel
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, type=str, help="The file containing the samples"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="If present, final code repair results will be written to this dataset.",
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
    parser.add_argument("--n", required=False, default=5, type=int)
    parser.add_argument(
        "--temperature",
        required=False,
        default=0.5,
        type=float,
        help="The temperature to use for the completion model",
    )
    parser.add_argument("--max-tokens", required=False, default=512, type=int)
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="The model to use for repair completions",
        choices=["codellama", "chatgpt", "gpt4"],
    )
    parser.add_argument(
        '--num-threads',
        required=False,
        type=int,
        default=2,
        help='The number of threads to use for parallelizing *on the task level*.',
    )
    parser.add_argument(
        '--stop-if-at',
        required=False,
        type=int,
        default=None,
    )
    return parser


def repair_code(
    problem,
    spec,
    code,
    model,
    explanation=None,
    breadth=5,
    temperature=0.8,
    max_tokens=256,
    parent_logger=None,
):
    
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

    if logger is not None:
        logger.add_log(
            "repair_code",
            "starting with args: " + json.dumps(args),
            log_level=LogLevel.DEBUG,
        )
    
    result = check_correctness(problem, code, 3.0)
    assert not result['passed'], f'Code {code} passed on problem {problem}.'
    instruction = f"The code does not pass the test cases. The error encountered was: `{result['result']}`"
    completions, prompt = model.generate_repair_completions(
        spec,
        code,
        instruction,
        explanation=explanation,
        temperature=temperature,
        max_tokens=max_tokens,
        logger=logger,
        num_samples=breadth,
    )
    code_generated = [model.extract_code_from_completion(c) for c in completions]
    return completions, code_generated, prompt, instruction


def get_model(s):
    # Load the model
    if s == "chatgpt":
        return GPT3_5()
    elif s == 'gpt4':
        return GPT4()
    elif s == 'codellama':
        return CodeLlamaVLLMAPI()
    else:
        raise ValueError(f"Unknown model {s}")


def main(args):

    he_problems = read_problems()['test']

    if args.logfile:
        logfile = args.logfile
    else:
        import datetime

        current_time = datetime.datetime.now()
        logfile = (
            "repair_log"
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
            p = j["task_id"]
            assert p not in ps, (p, ps[p], j)
            ps[p] = j

    length_before = len(ps)

    if args.output and os.path.exists(args.output):
        count = {}
        with open(args.output, "r") as f:
            for line in f.readlines():
                line = line.strip()
                j = json.loads(line)
                if len(j.keys()) == 1:
                    p = list(j.keys())[0]
                else:
                    p = j["task_id"]
                assert p in ps, f"problem {p} not in input file but in output file"
                count[p] = count.get(p, 0) + max(len(c["repairs"] or []) for c in j["completions"])
                if (all(c["binary"] for c in j["completions"])
                    or (args.stop_if_at is not None and count[p] >= args.stop_if_at)
                ):
                    logger.add_log_flush(
                        "main",
                        f"Skipping problem {p} because all its completions are done",
                        log_level=LogLevel.INFO,
                    )
                    ps.pop(p)

    length_after = len(ps)
    logger.add_log_flush(
        "main",
        f"filtered out {length_before - length_after} problems that were already in the output file",
        log_level=LogLevel.INFO,
    )

    logger.add_log_flush(
        "main",
        f">>> running on {len(ps)} problems",
        log_level=LogLevel.INFO,
    )


    model = get_model(args.model)

    def do_repair(problem_json):
        task_id = problem_json["task_id"]
        problem = he_problems[task_id]
        jsons = problem_json["completions"]
        local_logger = AtomicLogger(logger.logfile, prints=False, level=logger.log_level)
        local_logger.add_log(
            "main",
            f"trying to repair {len(jsons)} entries for task_id {problem}",
            log_level=LogLevel.INFO,
        )

        for j in jsons:
            if isinstance(j["binary"], str):
                j["binary"] = j["binary"] == "True"  # for legacy reasons

        for j in jsons:
            j["repairs"] = []

        if any(not j["binary"] for j in jsons):
            local_logger.add_log(
                "main",
                f"repairing {len(list(j for j in jsons if not j['binary']))} entries for task_id {problem}",
                log_level=LogLevel.INFO,
            )
            # execute all the base programs in parallel
            i_code_expl = []
            for i, j in enumerate(jsons):
                if not j["binary"]:
                    if j.get("explanations", None) is None:
                        i_code_expl.append((i, j["executed_completion"], None))
                    else:
                        for explanation in j["explanations"]:
                            i_code_expl.append((i, j["executed_completion"], explanation))
            to_exec = [] # list of (json, repairs) pairs to execute after getting all repairs
            # generate all the repairs
            bar = tqdm(total=len(i_code_expl), desc="Repairs generated")
            with ThreadPoolExecutor() as p:
                future_to_idx = {
                    p.submit(
                        repair_code,
                        problem,
                        problem['prompt'],
                        clean_code,
                        model,
                        explanation=explanation,
                        breadth=args.n,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        parent_logger=local_logger,
                    ): (i, explanation)
                    for i, clean_code, explanation in i_code_expl
                }
            for future in as_completed(future_to_idx):
                i, expl = future_to_idx[future]
                repairs = future.result()
                to_exec.append((i, repairs, expl))
                bar.update(n=1)
            # execute all the repairs in parallel
            bar = tqdm(total=sum(len(repairs[1]) for _, repairs, _ in to_exec), desc="Repairs executed")
            with ThreadPoolExecutor() as p:
                future_to_data = {
                    p.submit(
                        check_correctness,
                        problem,
                        code,
                        3.0): (i, completion, code, repairs[2], repairs[3], expl)
                    for i, repairs, expl in to_exec
                    for completion, code in zip(repairs[0], repairs[1])
                }
            for future in as_completed(future_to_data):
                i, completion, code, prompt, instruction, explanation = future_to_data[future]
                if not explanation:
                    # if no pre-existing explanation, use the completion up until the code
                    explanation = completion.split(model.open_delim)[0].rstrip()
                res = future.result()
                jsons[i]["repairs"].append(
                    {
                        "repaired_program": code,
                        "explanation": explanation,
                        "binary": res['passed'],
                        "error_msg": instruction,
                        "prompt": prompt,
                        "code": code,
                    }
                )
                bar.update(n=1)
        else:
            local_logger.add_log(
                "main",
                f"no need to repair {len(jsons)} entries for task_id {problem}",
                log_level=LogLevel.INFO,
            )

        problem_json["completions"] = jsons
        local_logger.add_log_flush(
            "main",
            f"finished repairing {len(jsons)} entries for task_id {problem}",
            log_level=LogLevel.INFO,
        )
        if args.output:
            with open(args.output, "a") as f:
                f.write(json.dumps(problem_json) + "\n")


    bar = tqdm(total=len(ps), desc="Problems handled")
    if args.num_threads == 1:
        for _, jsons in ps.items():
            do_repair(jsons)
            bar.update(n=1)

    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as p:
            fs = [
                p.submit(do_repair, problem_json)
                for (_, problem_json) in ps.items()
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
            f"Produced dataset of repaired programs written to {args.output}.",
            log_level=LogLevel.INFO,
        )


if __name__ == "__main__":
    main(build_args().parse_args())
