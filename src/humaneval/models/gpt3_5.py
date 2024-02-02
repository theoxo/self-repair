from models.OAIModel import OAIModel

class GPT3_5(OAIModel):
    MODEL_NAME = "gpt-3.5-turbo-0301"

    def __init__(self):
        super().__init__(GPT3_5.MODEL_NAME, open_delim='```python', close_delim='```')