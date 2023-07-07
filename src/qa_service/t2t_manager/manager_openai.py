"""
This file was created by ]init[ AG 2023.

Module for Text Generation Models from Open AI via API (GPT).
"""
import logging
import os

# pip install openai
# export OPENAI_API_KEY=sk-........
import openai
import threading

# With tiktoken as tokenizer:
# pip install tiktoken
import tiktoken
from timeit import default_timer as timer

# With transformer.GPT2TokenizerFast as tokenizer:
# from transformers import GPT2TokenizerFast

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class T2tManager:
    lock = threading.Lock()

    def __new__(cls):
        # Singleton!
        if not hasattr(cls, "instance"):
            cls.instance = super(T2tManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, "tokenizer"):
            return
        with T2tManager.lock:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            # Load tokenizer for counting tokens (Use tiktoken or transformer.GPT2TokenizerFast)
            # With tiktoken as tokenizer:
            self.tokenizer = tiktoken.get_encoding("gpt2")
            # With transformer.GPT2TokenizerFast as tokenizer:
            # model_id = 'gpt2'
            # models_folder = os.environ.get('MODELS_FOLDER', '/opt/speech_service/models/')
            # log.info(f"Loading tokenizer {model_id!r} in folder {models_folder!r}...")
            # self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id, cache_dir=models_folder)
            # log.info("...done.")

    def get_max_sequence_length(self) -> int:
        return 4000  # text-davinci-003

    def answer(self, question: str, context: str, top_k: int = 5) -> list:
        log.debug(f"Answering...")
        start = timer()

        prompt = f"Frage: {question}\nFakten:\n-----\n{context}"
        prompt_end = "\n-----\nKurze Antwort (vollständige Aufzählung): "

        # Shorten prompt for max sequence length 4000:
        max_length = self.get_max_sequence_length()
        while len(self.tokenizer.encode(prompt + prompt_end)) >= max_length:
            prompt = prompt.rsplit("\n", 1)[0]

        prompt += prompt_end
        log.debug(f"Prompt:\n{prompt}")

        answer = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=500, temperature=0.7)
        print(answer)

        answers = answer["choices"]  # type: ignore

        log.debug(f"...done in {timer() - start:.3f}s")
        assert isinstance(answers, list)
        return answers
