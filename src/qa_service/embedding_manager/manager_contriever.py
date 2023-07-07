"""
This file was created by ]init[ AG 2023.

Module for Sentence Embedding Models.

This is a test for the model 'facebook/mcontriever-msmarco', which is not part of SentenceTransformer.
To use it, change import in '__init__.py'!
This model is quite slow and not much better than other SentenceTransformers.
"""
import logging
import os
import sys
import threading
from timeit import default_timer as timer
import torch
from transformers import AutoTokenizer, pipeline

# cd /opt
# git clone https://github.com/facebookresearch/contriever.git
sys.path.append(os.path.abspath("/opt/contriever"))
from src.contriever import Contriever

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class EmbeddingManager:
    lock = threading.Lock()

    def __new__(cls):
        # Singleton!
        if not hasattr(cls, "instance"):
            cls.instance = super(EmbeddingManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, "model"):
            return
        with EmbeddingManager.lock:
            # Load model for Sentence Embedding

            model_id = "facebook/mcontriever-msmarco"  # Model Size is around 0.7 GB
            # Max token length is 512 -> Embedding is 768 dimensional
            # This model is good, small and just 768-dim, but it's really slow and memory intensive - 5/10

            models_folder = os.environ.get("MODELS_FOLDER", "/opt/speech_service/models/")
            device = "cpu"
            if torch.cuda.is_available():
                log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                mem_info = torch.cuda.mem_get_info(0)
                vram = round(mem_info[1] / 1024**3, 1)
                log.info(f"VRAM available: {round(mem_info[0]/1024**3,1)} GB out of {vram} GB")
                if vram >= 4:
                    device = "cuda:0"
            log.info(f"Loading model {model_id!r} in folder {models_folder!r}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=models_folder)
            self.model = Contriever.from_pretrained(model_id, cache_dir=models_folder).to(device)
            log.info(f"Max Sequence Length: {self.get_max_sequence_length()}")
            log.info(f"Embedding Dimensions: {self.get_embedding_dimensions()}")
            log.info("...done.")
            if device != "cpu":
                log.info(f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")

    def get_max_sequence_length(self) -> int:
        return self.model.config.max_position_embeddings

    def get_embedding_dimensions(self) -> int:
        return self.model.config.pooler_fc_size

    def embed(self, sentences: list[str]) -> torch.Tensor:
        log.debug(f"Embedding {len(sentences)} sentences...")
        start = timer()

        # Memory issues - needs batching:
        # inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        # embeddings = self.model(**inputs)
        # Better use transformer pipeline:

        embedding_pipeline = pipeline(
            "feature-extraction",
            self.model,
            tokenizer=self.tokenizer,
            device=self.model.device,
            max_length=512,
            return_tensors=True,
        )
        embeddings = embedding_pipeline(sentences)
        embeddings = torch.stack(embeddings)  # type: ignore

        log.debug(f"...done in {timer() - start:.3f}s")
        assert isinstance(embeddings, torch.Tensor)
        return embeddings
