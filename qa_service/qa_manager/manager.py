'''
This file was created by ]init[ AG 2023.

Module for Sentence Embedding Models.
'''
import logging
import os
import threading
from timeit import default_timer as timer
import torch
from transformers import pipeline

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
        if hasattr(self, 'pipeline'):
            return
        with QaManager.lock:
            # Load model for Question Answering (QA)

            model_id = 'Sahajtomar/German-question-answer-Electra'  # Model Size is around 1.24 GB

            # model_id = 'deepset/gelectra-large-germanquad'  # Model Size is around 1.3 GB
            # Max sequence length is 512
            # This model is really good, good with rivers, bad with mayor - 9/10

            # model_id = 'svalabs/rembert-german-question-answering'  # Model Size is around 2.16 GB

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
                'question-answering',
                model=model_id,
                tokenizer=model_id,
                device=device,
                model_kwargs={
                    'cache_dir': models_folder
                })
            log.info("...done.")
            if device != 'cpu':
                log.info(f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")

    def answer(self, question: str, context: str, top_k: int = 5) -> list:
        log.debug(f"Answering...")
        start = timer()
        answers = self.pipeline({
            'question': question,
            'context': context
        }, top_k=top_k)
        log.debug(f"...done in {timer() - start:.3f}s")
        assert isinstance(answers, list)
        return answers
