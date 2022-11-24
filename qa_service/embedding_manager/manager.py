'''
This file was created by ]init[ AG 2022.

Module for Sentence Embedding Models.
'''
import logging
import os
from sentence_transformers import SentenceTransformer
import threading
from timeit import default_timer as timer
import torch

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class EmbeddingManager:

    lock = threading.Lock()

    def __new__(cls):
        # Singleton!
        if not hasattr(cls, 'instance'):
            cls.instance = super(EmbeddingManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, 'sentence_splitters'):
            return
        with EmbeddingManager.lock:
            # Load model Sentence Embedding
            # Max sequence length is 512 -> embedding is 1024 dimensional
            model_id = 'Sahajtomar/German-semantic'
            model_folder = os.environ.get('MODEL_FOLDER', '/opt/speech_service/models/')
            device = 'cpu'
            if torch.cuda.is_available():
                log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                mem_info = torch.cuda.mem_get_info(0)
                vram = round(mem_info[1] / 1024**3, 1)
                log.info(
                    f"VRAM available: {round(mem_info[0]/1024**3,1)} GB out of {vram} GB")
                if (vram >= 4):
                    device = 'cuda:0'
                    # model_id = 'facebook/nllb-200-3.3B' if vram >= 32 else 'facebook/nllb-200-distilled-1.3B' if vram >= 12 else 'facebook/nllb-200-distilled-600M'
            log.info(f"Loading model {model_id!r} in folder {model_folder!r}...")
            self.model = SentenceTransformer(model_id, device=device, cache_folder=model_folder)
            log.info("...done.")
            if device != 'cpu':
                log.info(f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")

    def embed(self, sentences: list[str]) -> torch.Tensor:
        log.debug(f"Embedding {len(sentences)} sentences...")
        start = timer()
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        log.debug(f"...done in {timer() - start:.3f}s")
        assert isinstance(embeddings, torch.Tensor)
        return embeddings
