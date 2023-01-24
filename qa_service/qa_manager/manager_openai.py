'''
This file was created by ]init[ AG 2023.

Module for Sentence Embedding Models.
'''
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


class QaManager:

    lock = threading.Lock()

    def __new__(cls):
        # Singleton!
        if not hasattr(cls, 'instance'):
            cls.instance = super(QaManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, 'tokenizer'):
            return
        with QaManager.lock:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            # Load tokenizer for counting tokens (Use tiktoken or transformer.GPT2TokenizerFast)
            # With tiktoken as tokenizer:
            self.tokenizer = tiktoken.get_encoding('gpt2')
            # With transformer.GPT2TokenizerFast as tokenizer:
            # model_id = 'gpt2'
            # models_folder = os.environ.get('MODELS_FOLDER', '/opt/speech_service/models/')
            # log.info(f"Loading tokenizer {model_id!r} in folder {models_folder!r}...")
            # self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id, cache_dir=models_folder)
            # log.info("...done.")

    def answer(self, question: str, context: str, top_k: int = 5) -> list:
        log.debug(f"Answering...")
        start = timer()

        prompt = f"Beantworte die folgende Frage kurz und nutze dabei ausschließlich die nachfolgend aufgelisteten Fakten:\nFrage:{question}\nFakten:\n-----\n{context}"

        # Shorten prompt for context window size (max. 4000, for wiggling space 3000)
        # With transformer.GPT2TokenizerFast as tokenizer:
        # while len(self.tokenizer(prompt)['input_ids']) > 3000:
        # With tiktoken as tokenizer:
        while len(self.tokenizer.encode(prompt)) > 3000:
            prompt = prompt.rsplit('\n', 1)[0]
        prompt += "\n-----\nAntwort:\n"

        answers = [prompt]

        answer = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7)
        print(answer)

        answers = answer['choices']  # type: ignore

        log.debug(f"...done in {timer() - start:.3f}s")
        assert isinstance(answers, list)
        return answers
