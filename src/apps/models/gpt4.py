from models.OAIModel import OAIModel

class GPT4(OAIModel):
    MODEL_NAME = "gpt-4-0314"

    def __init__(self):
        super().__init__(GPT4.MODEL_NAME, open_delim='```python', close_delim='```')