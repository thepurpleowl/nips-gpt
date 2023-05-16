#%%
from jinja2 import Environment
import tiktoken

TOKENIZER_MODEL_ALIAS_MAP = {
    "gpt-4": "gpt-3.5-turbo",
    "gpt-35-turbo": "gpt-3.5-turbo",
}


def count_tokens(input_str: str, model_name: str) -> int:
    if model_name in TOKENIZER_MODEL_ALIAS_MAP:
        model_name = TOKENIZER_MODEL_ALIAS_MAP[model_name]
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens, encoding

class BugBasherSimplePromptConstructor:
    def __init__(
        self,
        template_path: str,
        model: str,
        max_length: int=4000,
        context_block_seperator: str='',
    ):
        
        with open(template_path, "r") as f:
            self.template = f.read()
        self.token_counter = lambda s: count_tokens(s, model_name=model)
        self.max_length = max_length
        self.context_block_seperator = context_block_seperator

        self._env = Environment()

    def construct(
        self,
        **kwargs,
    ):
        template = self._env.from_string(self.template)
        prompt = template.render(
            **kwargs,
        )
        prompt_len, _ = self.token_counter(prompt)
        remaining = self.max_length - prompt_len

        if remaining < 0:
            # raise ValueError(f"Context length exceeded. Max allowed context length is {self.max_length} tokens.")
            return None

        return prompt
