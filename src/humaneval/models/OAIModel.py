from time import sleep
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from AtomicLogger import LogLevel

class OAIModel:

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

    def get_generation_context(self):
        return [
            {
                "role": "system",
                "content": f"You are an expert Python programmer. Given the first few lines of a Python program, your task is to generate a complete implementation of the program. Put your program within code delimiters, for example: {self.open_delim}\n# YOUR CODE HERE\n{self.close_delim}.",
            },
        ]

    def get_explanation_context(self):
        return [
            {
                "role": "system",
                "content": "You are a helpful programming assistant. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by giving a detailed but concise textual explanation of what is wrong with the code. You will *not* generate any code, because the user wants to fix the code themselves.",
            },
            {
                "role": "user",
                "content": open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'joint_explanation_user.txt'), 'r').read().replace('$0$', self.open_delim).replace('$1$', self.close_delim),
            },
            {
                "role": "assistant",
                "content": open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'explanation_assistant.txt'), 'r').read().replace('$0$', self.open_delim).replace('$1$', self.close_delim),
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
                "content": open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'joint_explanation_user.txt'), 'r').read().replace('$0$', self.open_delim).replace('$1$', self.close_delim),
            },
            {
                "role": "assistant",
                "content": open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'joint_explanation_assistant.txt'), 'r').read().replace('$0$', self.open_delim).replace('$1$', self.close_delim),
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
                "content": open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repair_user.txt'), 'r').read().replace('$0$', self.open_delim).replace('$1$', self.close_delim),
            },
            {
                "role": "assistant",
                "content": open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repair_assistant.txt'), 'r').read().replace('$0$', self.open_delim).replace('$1$', self.close_delim),
            },
        ]

    def construct_repair_prompt(self, spec, incorrect_code, error_message, explanation):
        context = self.get_repair_context()
        context.append(
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "### QUESTION",
                        spec,
                        "### INCORRECT CODE",
                        self.open_delim,
                        incorrect_code,
                        self.close_delim,
                        error_message,
                        explanation,
                    ]
                ),
            },
        )
        return context

    def construct_explanation_prompt(self, spec, incorrect_code, error_message):
        context = self.get_explanation_context()
        context.append(
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "### QUESTION",
                        spec,
                        "### INCORRECT CODE",
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
                        "### QUESTION",
                        spec,
                        "### INCORRECT CODE",
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
        if "### QUESTION" in completion:
            completion = completion.split("### QUESTION")[0]
        if "[PYTHON]" in completion:
            completion = completion.split("[PYTHON]")[1].split("[/PY")[0]
        if "```python" in completion:
            completion = completion.split("```python")[1].split("```")[0]
        elif "```" in completion:
            completion = completion.split("```")[1]
        return completion