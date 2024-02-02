from models.OAIModel import OAIModel

class CodeLlamaVLLMAPI(OAIModel):
    MODEL_NAME = "codellama/CodeLlama-13b-Instruct-hf"

    def __init__(self):
        address = "http://localhost:8889/v1"
        super().__init__(CodeLlamaVLLMAPI.MODEL_NAME, open_delim='[PYTHON]', close_delim='[/PYTHON]', openai_base=address)

    def preprocess_prompt(self, prompt):
        # converting chat format to llama instruction format
        system_msg = prompt[0]["content"]
        texts = [f'<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n']
        for user_msg, response in zip(prompt[1::2], prompt[2::2]):
            user_msg = user_msg["content"]
            response = response["content"]
            texts.append(f'{user_msg.strip()} [/INST] {response.strip()} </s><s>[INST] ')
        texts.append(f'{prompt[-1]["content"].strip()} [/INST]')
        prompt = ''.join(texts)
        return prompt
