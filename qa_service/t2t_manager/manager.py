'''
This file was created by ]init[ AG 2023.

Module for Text Generation Models.
'''
import os
# pip install bitsandbytes accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
import threading
from timeit import default_timer as timer
import torch
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
            # Max sequence length is 512

            # model_id = 'google/flan-t5-xxl'  # Model Size is around 50 GB
            # Meeds at least 24 GB VRAM? -->
            # Exception has occurred: OutOfMemoryError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
            # CUDA out of memory. Tried to allocate 40.00 MiB (GPU 0; 15.99 GiB total
            # capacity; 14.85 GiB already allocated; 0 bytes free; 15.13 GiB reserved
            # in total by PyTorch) If reserved memory is >> allocated memory try
            # setting max_split_size_mb to avoid fragmentation.  See documentation for
            # Memory Management and PYTORCH_CUDA_ALLOC_CONF

            # model_id = 'google/mt5-xl'  # Model Size is around 13.9 GB
            # Needs: pip install protobuf==3.20.*
            # Isn't optimized for tasks like flan - more prompt engineering needed?!

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
            # tokenizer = T5Tokenizer.from_pretrained(model_id, cache_dir=models_folder)
            # model = T5ForConditionalGeneration.from_pretrained(
            #    model_id, cache_dir=models_folder, device_map='auto', load_in_8bit=True)
            # input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to('cuda')
            # outputs = model.generate(input_ids, max_new_tokens=500)  # type: ignore
            # response = tokenizer.decode(outputs[0])
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
        text: str = self.pipeline(prompt)  # type: ignore
        log.debug(f"...done in {timer() - start:.3f}s")
        # assert isinstance(answers, list)
        return text

    def answer(self, question: str, context: str) -> str:
        log.debug(f"Answering...")
        start = timer()

        prompt = f"Frage: {question}\nFakten:\n-----\n{context}"
        prompt_end = '\n-----\nKurzer Antwortsatz mit allen passenden Fakten: '

        # Shorten prompt for max sequence length 512:
        max_length = self.get_max_sequence_length()
        while len(
            self.pipeline.tokenizer(
                prompt + prompt_end,
                return_tensors='pt').input_ids[0]) >= max_length:  # type: ignore
            prompt = prompt.rsplit('\n', 1)[0]

        prompt += prompt_end
        log.debug(f"Prompt:\n{prompt}")

        response: list[dict] = self.pipeline(prompt, max_length=100)  # type: ignore
        log.debug(f"...done in {timer() - start:.3f}s")
        return response[0]['generated_text']
