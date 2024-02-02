import json
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pebble import (
    ProcessPool,
)  # timeout support seems a bit busted in concurrent.futures
from tqdm import tqdm
from colorama import Fore
from concurrent.futures import TimeoutError, ThreadPoolExecutor, as_completed
import numpy as np
from AtomicLogger import LogLevel

APPS_DIR = os.environ["APPS_DIR"]


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, type=str, help="The file to read the input from"
    )
    parser.add_argument(
        "--output", required=True, type=str, help="The file to write the output to"
    )
    parser.add_argument(
        "--key",
        required=False,
        default="executed_completion",
        type=str,
        help="The key to use for the completion",
    )
    return parser


def check_pass(errors):
    return all(isinstance(s, str) and s == "passed" for s in errors)

def exec_sample(prob_path, completion, logger=None):
    """
    This function executes the given completion against the unit tests for the corresponding problem.
    It may also pre-process the completion to, for example, remove trailing whitespace
    (not typically needed with contemporary LLMs).
    The first item in the returned tuple should be a list of test case results, one per test,
    with each entry being a tuple (type of error, error message).
    :return a tuple (error or success, processed_completion)
    """
    raise NotImplementedError(
            "The source code for this function could not be released. "
            "It must be implemented before the codebase can be used for APPS.")

def wrap_strings_truncate(x, max_len=300):
    if isinstance(x, str):
        if len(x) == 0 or x[0] != '"':
            x = f'"{x}'
        if len(x) == 0 or x[-1] != '"':
            x = f'{x}"'
    if hasattr(x, "__len__") and len(x) > max_len:
        x = str(x[: max_len - 4]) + "..." + str(x[-1])
    return x


def get_error_message(prob_path, errors, logger=None, include_line_numbers=True):
    instr = None

    if len(errors) == 0:
        if logger is not None:
            logger.add_log(
                "get_error_message", f"Zero errors for {prob_path}", LogLevel.WARNING
            )
        raise Exception("Zero errors for {prob_path}")

    assert not all(
        isinstance(x, str) and x == "passed" for x in errors
    ), f"All errors are strings for {prob_path}"
    for err_idx, err in enumerate(errors):
        if isinstance(err, str):
            assert err == "passed", f'error "{err}" was string but not "passed"'
        else:
            assert isinstance(err, tuple) or isinstance(err, list)
            err_type = err[0]
            if err_type == "FUNCTION NOT FOUND ERROR":
                instr = f'The code above is missing a function. The error message was "{err[1]}".'
                break
            elif err_type == "COMPILE ERROR":
                err_type = err[1][: err[1].find("(")]
                if include_line_numbers and len(err) > 2:
                    err_type += f" on line {err[2]}"
                instr = f'The code above has a {err_type}. The error message was "{err[1]}". '
                break
            elif err_type == "RUNTIME ERROR":
                err_type = err[1][: err[1].find("(")]
                if include_line_numbers and len(err) > 2:
                    err_type += f" on line {err[2]}"
                if len(err) > 3:
                    input = err[-1]
                    instr = f'The code above has a {err_type} when given input {wrap_strings_truncate(input)}. The error message was "{err[1]}".'
                else:
                    instr = f'The code above has a {err_type}. The error message was "{err[1]}".'

            else:
                assert err_type == "WRONG ANSWER", f'Unexpected err_type "{err_type}"'
                outputs = err[1]
                expected_outputs = err[2]
                input = None
                with open(
                    os.path.join(
                        prob_path,
                        "input_output.json",
                    ),
                    "r",
                ) as f:
                    input_output = json.load(f)
                    input = input_output["inputs"][
                        err_idx
                    ]  # assuming inputs are executed, and results are returned, in order
                assert (
                    input is not None
                ), f"Could not find input for expected output {expected_outputs}, prob_path {prob_path}"

                def recurse_flatten(x):
                    if isinstance(x, list) or isinstance(x, tuple):
                        if len(x) == 1:
                            return recurse_flatten(x[0])
                        else:
                            return [recurse_flatten(y) for y in x]
                    else:
                        return x
                
                def recurse_join(x):
                    if isinstance(x, list) or isinstance(x, tuple):
                        if all([isinstance(y, str) for y in x]):
                            return " ".join(x)
                        else:
                            return [recurse_join(y) for y in x]
                    else:
                        return x

                expected_outputs = recurse_flatten(expected_outputs)
                expected_outputs = recurse_join(expected_outputs)
                outputs = recurse_flatten(outputs)
                outputs = recurse_join(outputs)

                if outputs == [] and not isinstance(expected_outputs, list):
                    outputs = None

                if (
                    (
                        isinstance(expected_outputs, list)
                        or isinstance(expected_outputs, tuple)
                    )
                    and (isinstance(outputs, list) or isinstance(outputs, tuple))
                    and (
                        len(expected_outputs) != len(outputs)
                        or any([x != y for x, y in zip(expected_outputs, outputs)])
                    )
                ) or outputs != expected_outputs:
                    instr = (
                        f"The code above is wrong and contains a bug. "
                        f"Given input {wrap_strings_truncate(input)} the output "
                        f"was {wrap_strings_truncate(outputs)} but the expected "
                        f"output is {wrap_strings_truncate(expected_outputs)}."
                    )
                    break
                else:
                    # this is a rare case that happens when the only mistake in the error is that
                    # it returns a singe-element list instead of a single element.
                    # In this case, we try to set the instr using the original outputs,
                    # but don't break so that the instruction will be overriden if possible
                    instr = (
                        f"The code above is wrong and contains a bug. "
                        f"Given input {wrap_strings_truncate(input)} the output "
                        f"was {wrap_strings_truncate(err[1])} but the expected "
                        f"output is {wrap_strings_truncate(err[2])}."
                    )

    if instr is None:
        if logger is not None:
            logger.add_log(
                "get_error_message",
                f"Could not generate error message for {prob_path}; errors were {errors}",
                LogLevel.ERROR,
            )
        raise Exception(
            f"Could not generate error message for {prob_path}; errors were {errors}"
        )
    if logger is not None:
        logger.add_log(
            "get_error_message",
            f"Generated error message for {prob_path}: {instr}",
            LogLevel.DEBUG,
        )

    return instr


def unbiased_estimate_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    from the Codex paper
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def comment(text):
    return "\n".join(
        map(
            lambda l: "# " + l if (len(l) > 0 and l[0] != "#") else l,
            text.strip().split("\n"),
        )
    )


def get_gold_standards(prob_path, cache, logger=None):
    if logger is not None:
        logger.add_log(
            "get_gold_standards",
            f"Getting gold standards for {prob_path}",
            log_level=LogLevel.DEBUG,
        )
    if prob_path in cache:
        if logger is not None:
            logger.add_log(
                "get_gold_standards",
                f"Cache hit for {prob_path}",
                log_level=LogLevel.DEBUG,
            )
    else:
        if logger is not None:
            logger.add_log(
                "get_gold_standards",
                f"Cache miss for {prob_path}",
                log_level=LogLevel.DEBUG,
            )
        with open(prob_path + "/solutions.json", "r") as f:
            gold_standards = json.load(f)
        if logger is not None:
            logger.add_log(
                "get_gold_standards",
                f"Loaded {len(gold_standards)} gold standards for {prob_path}: {gold_standards}",
                log_level=LogLevel.DEBUG,
            )
        cache[prob_path] = gold_standards

    return cache[prob_path]


def is_call_based(prob_path, logger=None):
    # Check if this problem excepts a call-based (or stdin/stdout-based) solution
    with open(os.path.join(prob_path, "input_output.json"), "r") as f:
        j = json.load(f)
    call_based = "fn_name" in j.keys()
    if logger is not None:
        infix = "not " if not call_based else ""
        logger.add_log(
            "is_call_based", f"Problem {prob_path} is {infix}call-based", LogLevel.DEBUG
        )
    return "fn_name" in j.keys()


def get_starter_code(prob_path, logger=None):
    # Get the starter code for this problem
    f_path = os.path.join(prob_path, "starter_code.py")
    if not os.path.isfile(f_path):
        starter_code = None
    else:
        with open(f_path, "r") as f:
            starter_code = f.read()
    if logger is not None:
        infix = "not " if starter_code is not None else ""
        logger.add_log(
            "get_starter_code",
            f"Starter code for {prob_path} is {infix}None",
            LogLevel.DEBUG,
        )
    return starter_code


def get_tests(prob_path, logger=None, max_tests=5, max_len=50):

    def get_tests_call_based(j):
        fn_name = j['fn_name']
        inputs = [wrap_strings_truncate(x, max_len=max_len) for x in j['inputs']]
        outputs = [wrap_strings_truncate(x, max_len=max_len) for x in j['outputs']]
        assert len(inputs) == len(outputs)
        _tests = '\n'.join([f'assert {wrap_strings_truncate(fn_name, max_len=max_len)}({wrap_strings_truncate(inputs[i], max_len=max_len)}) == {wrap_strings_truncate(outputs[i], max_len=max_len)}' for i in range(len(inputs))])
        return '### UNIT TESTS' + '\n```python\n' + _tests + '\n```'

    def get_tests_stdio_based(j):
        inputs = [wrap_strings_truncate(x, max_len=max_len) for x in j['inputs']]
        outputs = [wrap_strings_truncate(x, max_len=max_len) for x in j['outputs']]
        assert len(inputs) == len(outputs)
        _tests = '\n'.join([f'## Test #{i}\n# Input\n{wrap_strings_truncate(inputs[i], max_len=max_len)}\n# Output\n{wrap_strings_truncate(outputs[i], max_len=max_len)}' for i in range(len(inputs))])
        return '### UNIT TESTS\n' + _tests + '\n'
    
    with open(os.path.join(prob_path, "input_output.json"), "r") as f:
        j = json.load(f)

    if len(j['inputs']) > max_tests:
        j['inputs'] = j['inputs'][:max_tests]
        j['outputs'] = j['outputs'][:max_tests]
        if logger is not None:
            logger.add_log(
                "get_tests",
                f"Truncated tests for {prob_path} to {max_tests}",
                LogLevel.DEBUG,
            )

    if "fn_name" in j.keys():
        res = get_tests_call_based(j)
    else:
        res = get_tests_stdio_based(j)

    if logger is not None:
        logger.add_log(
            "get_tests",
            f"Tests for {prob_path}: {res}",
            LogLevel.DEBUG,
        )
    return res


def get_difficulty(prob_path):
    with open(os.path.join(prob_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
        return metadata["difficulty"]


def main(args):
    with open(args.input, "r") as f:
        lines = [json.loads(x.strip()) for x in f.readlines()]
        lines = {x["prob_path"]: x for x in lines}

    with open(args.output, "r") as f:
        for line in f.readlines():
            j = json.loads(line.strip())
            p = j["prob_path"]
            lines.pop(p)

    def color(x):
        if x:
            c = Fore.GREEN
        else:
            c = Fore.RED
        return f"{c}{x}{Fore.RESET}"

    for _, line in tqdm(sorted(lines.items(), key=lambda kv: kv[0])):
        with ThreadPoolExecutor() as pool:
            future_to_compl = {
                pool.submit(exec_sample, line["prob_path"], x["original_completion"]): x
                for x in line["completions"]
            }
            for future in tqdm(
                as_completed(future_to_compl), total=len(future_to_compl)
            ):
                errors, exec_compl = future.result()
                compl = future_to_compl[future]

                binary = check_pass(errors)
                if binary and not compl["binary"]:
                    print(
                        f'Re-execution fixed a completion of {line["prob_path"]}: {color(binary)} <- {color(compl["binary"])}'
                    )
                    compl["binary"] = binary
                    compl["executed_completion"] = exec_compl
                    compl["fault"] = "passed"

        with open(args.output, "a") as f:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main(build_args().parse_args())
