'''
This file was created by ]init[ AG 2023.

Module for Text Generation Models.
'''
import os
import logging
import threading
from timeit import default_timer as timer
import torch
# pip install bitsandbytes accelerate
from transformers import pipeline

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class T2tManager:

    lock = threading.Lock()

    def __new__(cls):
        # Singleton!
        if not hasattr(cls, 'instance'):
            cls.instance = super(T2tManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, 'pipeline'):
            return
        with T2tManager.lock:
            # Load model for Text-to-Text Generation

            model_id = 'google/flan-t5-xl'  # Model Size is around 10.6 GB
            # Text2TextGeneration
            # Max sequence length is 512

            # model_id = 'google/flan-t5-xxl'  # Model Size is around 41.9 GB
            # Text2TextGeneration
            # Meeds at least 24 GB VRAM? -->

            # model_id = 'google/flan-ul2'
            # Text2TextGeneration

            # model_id = 'bigscience/bloomz-7b1-mt'  # Model Size is around 13.1 GB
            # TextGeneration: Isn't instruction tuned, just answers 'Yes'
            # Max sequence length is 512

            # model_id = 'google/mt5-xl'  # Model Size is around 13.9 GB
            # Text2TextGeneration: Isn't instruction tuned
            # Needs: pip install protobuf==3.20.*
            # Tokenization seems defect with pipelines...

            models_folder = os.environ.get('MODELS_FOLDER', '/opt/speech_service/models/')

            device = 'cpu'
            if torch.cuda.is_available():
                log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                mem_info = torch.cuda.mem_get_info(0)
                vram = round(mem_info[1] / 1024**3, 1)
                log.info(
                    f"VRAM available: {round(mem_info[0]/1024**3,1)} GB out of {vram} GB")
                if (vram >= 4):
                    device = 'cuda:0'
            log.info(f"Loading model {model_id!r} in folder {models_folder!r}...")
            self.pipeline = pipeline(
                'text2text-generation',
                model=model_id,
                tokenizer=model_id,
                model_kwargs={
                    'cache_dir': models_folder,
                    'device_map': 'auto',
                    'load_in_8bit': True
                })
            log.info(f"Max Sequence Length: {self.get_max_sequence_length()}")
            log.info("...done.")
            if device != 'cpu':
                log.info(f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")

    def get_max_sequence_length(self) -> int:
        return self.pipeline.tokenizer.model_max_length  # type: ignore

    def generate(self, prompt: str) -> str:
        log.debug(f"Generating...")
        start = timer()
        response: list[dict] = self.pipeline(prompt, max_length=1024)  # type: ignore
        log.debug(f"...done in {timer() - start:.3f}s")
        # assert isinstance(answers, list)
        return response[0]['generated_text']

    def answer(self, question: str, context: str) -> str:
        log.debug(f"Answering...")
        start = timer()

        prompt = f"Frage: {question}\nFakten:\n-----\n{context}"
        prompt_end = '\n-----\nKurze Antwort (vollständige Aufzählung): '

        # T5 was trained for max sequence length 512, but can handle longer sequences!
        # There will be a warning and the model might not attend to all facts, but it works.
        # Shorten prompt for max sequence length 512:
        # max_length = self.get_max_sequence_length()
        # while len(
        #     self.pipeline.tokenizer(
        #         prompt + prompt_end,
        #         return_tensors='pt').input_ids[0]) >= max_length:  # type: ignore
        #     prompt = prompt.rsplit('\n', 1)[0]

        prompt += prompt_end
        log.debug(f"Prompt:\n{prompt}")

        response: list[dict] = self.pipeline(prompt, max_new_tokens=200)  # type: ignore
        log.debug(f"...done in {timer() - start:.3f}s")
        return response[0]['generated_text']
