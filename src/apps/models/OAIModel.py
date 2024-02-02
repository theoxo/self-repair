from time import sleep
import json
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from AtomicLogger import LogLevel

APPS_DIR = os.environ["APPS_DIR"]

class OAIModel:

    QUESTION_DELIMITER = "### QUESTION"
    DEFAULT_CODE_DELIMITER = "### PYTHON CODE"
    FIXED_CODE_DELIMITER = "### FIXED PYTHON CODE"
    INCORRECT_CODE_DELIMITER = "### INCORRECT PYTHON CODE"

    CALL_BASED_SPEC = open(os.path.join(APPS_DIR, 'train/3177/question.txt'), 'r').read() + '\n' + '### Use Call-Based Format'
    CALL_BASED_WRONG_PROG = "def is_pal(s):\n    return s == s[::-1]\n\ndef palindrome(num):\n    if not isinstance(num, int) or num < 0:\n        return 'Not valid'\n    s = str(num)\n    pals = set()\n    for i, ch in enumerate(s):\n        for j in range(i + 2, len(s) + 1):\n            test = s[i:j]\n            if is_pal(test):\n                pals.add(test)\n    return sorted(int(x) for x in pals) or 'No palindromes found'"
    CALL_BASED_INSTRUCTION = 'The code above is wrong and contains a bug. Given input "1001331" the output was "[\'0\', \'33\', \'1001\', \'1331\']" but the expected output was "[\'33\', \'1001\', \'1331\']".'
    CALL_BASED_EXPLANATION = "The problem description states that numbers which start or end with zeros (such as `010` and `00`) are NOT considered valid numerical palindromes. However, the code above does not take this into account and therefore returns `00` as a valid palindrome. This can be fixed by checking if the first or last character is `0` before adding the string to the set of palindromes."
    CALL_BASED_CORRECT_PROG = "def is_pal(s):\n    return s == s[::-1]\n\ndef palindrome(num):\n    if not isinstance(num, int) or num < 0:\n        return 'Not valid'\n    s = str(num)\n    pals = set()\n    for i, ch in enumerate(s):\n        if ch == '0':\n            continue\n        for j in range(i + 2, len(s) + 1):\n            test = s[i:j]\n            if is_pal(test):\n                pals.add(test)\n    return sorted(int(x) for x in pals) or 'No palindromes found'"

    STDIO_BASED_SPEC = open(os.path.join(APPS_DIR, 'train/0000/question.txt'), 'r').read() + '\n' + '### Use Standard Input Format (read inputs with `input()`, write results with `print()`)'
    STDIO_BASED_CORRECT_PROG = json.load(open(os.path.join(APPS_DIR, 'train/0000/solutions.json'), 'r'))[0]

    EXPLANATION_PROMPT = (
        "The following is a concise explanation of the issue:"
    )
    END_INSTR = "# The code below is the correct version of the code above, where the issue has been fixed:"

    def __init__(self, model_name, open_delim='```python', close_delim='```', openai_base=None):
        self.model_name = model_name
        self.open_delim = open_delim
        self.close_delim = close_delim
        if openai_base is not None:
            openai.api_base = openai_base
    
    def is_chat_based(self):
        return 'gpt-4' in self.model_name or 'gpt-3.5' in self.model_name

    def get_model_name(self):
        return self.model_name

    def get_generation_context(self, call_based):
        q = OAIModel.CALL_BASED_SPEC if call_based else OAIModel.STDIO_BASED_SPEC
        p = OAIModel.CALL_BASED_CORRECT_PROG if call_based else OAIModel.STDIO_BASED_CORRECT_PROG
        return [
            {
                "role": "system",
                "content": f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Put your fixed program within code delimiters, for example: {self.open_delim}\n# YOUR CODE HERE\n{self.close_delim}.",
            },
            {
                "role": "user",
                "content": OAIModel.QUESTION_DELIMITER + "\n" + q
            },
            {
                "role": "assistant",
                "content": {self.open_delim} + p + "\n" + {self.close_delim}
            },
        ]

    def get_explanation_context(self):
        return [
            {
                "role": "system",
                "content": "You are a helpful programming assistant and an expert Python programmer. You are helping a user debug a program. The user has written some code, but it has some errors and is not passing the tests. You will help the user by giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. You will *not* generate any code, because the user wants to fix the code themselves.",
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        OAIModel.QUESTION_DELIMITER,
                        OAIModel.CALL_BASED_SPEC,
                        OAIModel.INCORRECT_CODE_DELIMITER,
                        self.open_delim,
                        OAIModel.CALL_BASED_WRONG_PROG,
                        self.close_delim,
                        OAIModel.CALL_BASED_INSTRUCTION,
                    ]
                ),
            },
            {
                "role": "assistant",
                "content": "\n".join(
                    [
                        OAIModel.EXPLANATION_PROMPT
                        + " "
                        + OAIModel.CALL_BASED_EXPLANATION,
                    ]
                ),
            },
        ]

    def get_joint_explanation_context(self):
        return [
            {
                "role": "system",
                "content": f"You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. Put your fixed program within code delimiters, for example: {self.open_delim}\n# YOUR CODE HERE\n{self.close_delim}.",
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        OAIModel.QUESTION_DELIMITER,
                        OAIModel.CALL_BASED_SPEC,
                        OAIModel.INCORRECT_CODE_DELIMITER,
                        self.open_delim,
                        OAIModel.CALL_BASED_WRONG_PROG,
                        self.close_delim,
                        OAIModel.CALL_BASED_INSTRUCTION,
                    ]
                ),
            },
            {
                "role": "assistant",
                "content": "\n".join(
                    [
                        OAIModel.EXPLANATION_PROMPT
                        + " "
                        + OAIModel.CALL_BASED_EXPLANATION,
                        OAIModel.END_INSTR,
                        OAIModel.FIXED_CODE_DELIMITER,
                        self.open_delim,
                        OAIModel.CALL_BASED_CORRECT_PROG,
                        self.close_delim,
                    ]
                ),
            },
        ]

    def get_repair_context(self):
        return [
            {
                "role": "system",
                "content": f"You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. The user has spent some time debugging the program and will provide you with a concise textual explanation of what is wrong with the code. You will use this explanation to generate a fixed version of the program. Put your fixed program within code delimiters, for example: {self.open_delim}\n# YOUR CODE HERE\n{self.close_delim}.",
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        OAIModel.QUESTION_DELIMITER,
                        OAIModel.CALL_BASED_SPEC,
                        OAIModel.INCORRECT_CODE_DELIMITER,
                        self.open_delim,
                        OAIModel.CALL_BASED_WRONG_PROG,
                        self.close_delim,
                        OAIModel.CALL_BASED_INSTRUCTION,
                        OAIModel.EXPLANATION_PROMPT
                        + " "
                        + OAIModel.CALL_BASED_EXPLANATION,
                    ]
                ),
            },
            {
                "role": "assistant",
                "content": "\n".join(
                    [
                        OAIModel.FIXED_CODE_DELIMITER,
                        self.open_delim,
                        OAIModel.CALL_BASED_CORRECT_PROG,
                        self.close_delim,
                    ]
                ),
            },
        ]

    def construct_repair_prompt(self, spec, incorrect_code, error_message, explanation):
        context = self.get_repair_context()
        context.append(
            {
                "role": "user",
                "content": "\n".join(
                    [
                        OAIModel.QUESTION_DELIMITER,
                        spec,
                        OAIModel.INCORRECT_CODE_DELIMITER,
                        self.open_delim,
                        incorrect_code,
                        self.close_delim,
                        error_message,
                        explanation,
                    ]
                ),
            }
        )
        return context

    def construct_explanation_prompt(self, spec, incorrect_code, error_message):
        context = self.get_explanation_context()
        context.append(
            {
                "role": "user",
                "content": "\n".join(
                    [
                        OAIModel.QUESTION_DELIMITER,
                        spec,
                        OAIModel.INCORRECT_CODE_DELIMITER,
                        self.open_delim,
                        incorrect_code,
                        self.close_delim,
                        error_message,
                    ]
                ),
            }
        )
        return context

    @staticmethod
    def construct_prompt(context, prompt):
        context.append({"role": "user", "content": prompt})
        return context

    def construct_explanation_and_repair_prompt(self, spec, incorrect_code, error_message):
        context = self.get_joint_explanation_context()
        context.append(
            {
                "role": "user",
                "content": "\n".join(
                    [
                        OAIModel.QUESTION_DELIMITER,
                        spec,
                        OAIModel.INCORRECT_CODE_DELIMITER,
                        self.open_delim,
                        incorrect_code,
                        self.close_delim,
                        error_message,
                    ]
                ),
            },
        )
        return context

    def generate_explanation_completions(
        self, question_spec, incorrect_code, error_message, logger=None, stop=['```python', '###'], **kwargs
    ):
        if logger is not None:
            logger.add_log(
                "generate_explanation_completions",
                f"question_spec={question_spec}, incorrect_code={incorrect_code}, error_message={error_message}",
                LogLevel.DEBUG,
            )
        prompt = self.construct_explanation_prompt(
            question_spec, incorrect_code, error_message
        )
        if logger is not None:
            logger.add_log(
                "generate_explanation_completions",
                f"prompt={prompt}",
                LogLevel.DEBUG,
            )
        return (self.get_completions(prompt, logger=logger, stop=stop, **kwargs), prompt)

    def generate_repair_completions(
        self, question_spec, incorrect_code, error_message, explanation=None, logger=None, **kwargs
    ):
        if logger is not None:
            logger.add_log(
                "generate_repair_completions",
                f"question_spec={question_spec}, incorrect_code={incorrect_code}, error_message={error_message}, explanation={explanation}",
                LogLevel.DEBUG,
            )
        if explanation is None:
            prompt = self.construct_explanation_and_repair_prompt(
                question_spec, incorrect_code, error_message
            )
        else:
            prompt = self.construct_repair_prompt(
                question_spec, incorrect_code, error_message, explanation
            )
        if logger is not None:
            logger.add_log(
                "generate_repair_completions",
                f"prompt={prompt}",
                LogLevel.DEBUG,
            )
        return (self.get_completions(prompt, logger=logger, **kwargs), prompt)

    def preprocess_prompt(self, prompt):
        return prompt  ### no-op by default, may be subclassed

    def get_completions(
        self,
        prompt,
        temperature=0.8,
        top_p=0.95,
        max_tokens=256,
        num_samples=1,
        stop="### QUESTION",
        echo=False,
        logger=None,
    ):
        if logger is not None:
            logger.add_log(
                "get_completions",
                f"n={num_samples}, temp={temperature}, max_tokens={max_tokens}, best_of={num_samples}, stop={stop}, echo={echo}",
                LogLevel.DEBUG,
            )

        prompt = self.preprocess_prompt(prompt)

        while True:
            try:
                if self.is_chat_based():
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=num_samples,
                        stop=stop,
                        messages=prompt,
                    )
                    gens = [c.message.content for c in response.choices]
                else:
                    response = openai.Completion.create(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=num_samples,
                        stop=stop,
                        top_p=top_p,
                    )
                    gens = [c.text for c in response.choices]
                if logger is not None:
                    logger.add_log(
                        "get_completions",
                        f"Got {len(gens)} completions; returning.", 
                        LogLevel.DEBUG,
                    )
                return gens
            except openai.error.Timeout:
                if logger is not None:
                    logger.add_log(
                        "get_completions",
                        f"Timeout. Sleeping for 10s.",
                        LogLevel.DEBUG,
                    )
                sleep(10)
            except openai.error.RateLimitError as e:
                if logger is not None:
                    logger.add_log(
                        "get_completions",
                        f"Rate limit reached. Sleeping for 1s. {e}",
                        LogLevel.DEBUG,
                    )
                sleep(1)
            except openai.error.APIError as e:
                if logger is not None:
                    logger.add_log(
                        "get_completions",
                        f"API error: {e}. Sleeping for 30s",
                        LogLevel.DEBUG,
                    )
                sleep(30)
            except openai.error.APIConnectionError as e:
                if logger is not None:
                    logger.add_log(
                        "get_completions",
                        f"API connection error: {e}. Sleeping for 10s",
                        LogLevel.DEBUG,
                    )
                sleep(10)
            except openai.error.InvalidRequestError as e:
                if "tokens" in str(e):
                    max_tokens = int(max_tokens * 0.75)
                    if logger is not None:
                        logger.add_log(
                            "get_completions",
                            f"Hit context window limit; retrying with 3/4={max_tokens} of the tokens",
                            LogLevel.WARNING,
                        )
                else:
                    if logger is not None:
                        logger.add_log(
                            "get_completions",
                            f"Invalid request: {e}",
                            LogLevel.ERROR,
                        )
                    raise e
            except Exception as ex:
                if logger is not None:
                    logger.add_log(
                        "get_completions",
                        f"Unknown error: {ex}",
                        LogLevel.ERROR,
                    )
                raise ex

    @staticmethod
    def extract_code_from_completion(completion):
        if OAIModel.QUESTION_DELIMITER in completion:
            completion = completion.split(OAIModel.QUESTION_DELIMITER, 1)[0]
        if OAIModel.FIXED_CODE_DELIMITER in completion:
            completion = completion.split(OAIModel.FIXED_CODE_DELIMITER)[1]
        if "[PYTHON]" in completion:
            completion = completion.split("[PYTHON]")[1].split("[/PY")[0]
        if "```python" in completion:
            completion = completion.split("```python")[1].split("```")[0]
        elif "```" in completion:
            completion = completion.split("```")[1]
        return completion