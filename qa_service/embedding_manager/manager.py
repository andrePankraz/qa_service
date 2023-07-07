'''
This file was created by ]init[ AG 2023.

Module for Sentence Embedding Models.
'''
import abc
import logging
import os
from sentence_transformers import SentenceTransformer
import threading
from timeit import default_timer as timer
import torch

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class AbstractEmbeddingManager(abc.ABC):

    _lock = threading.Lock()

    def __new__(cls):
        # Singleton!
        if not hasattr(cls, 'instance'):
            with cls._lock:
                if not hasattr(cls, 'instance'):
                    cls.instance = super().__new__(cls)
        return cls.instance

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def max_sequence_length(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def embedding_dimensions(self) -> int:
        pass

    @abc.abstractmethod
    def embed(self, paragraphs: list[str]) -> torch.Tensor:
        pass


class EmbeddingManagerOnPrem(AbstractEmbeddingManager):

    _lock = threading.Lock()

    def __init__(self) -> None:
        with EmbeddingManagerOnPrem._lock:
            if hasattr(self, '_initialized'):
                return
            # Load model for Paragraph Embedding (Bi-Encoder)

            # Model Size is around 1.3 GB, mostly German, but English works too,
            # Max sequence length is 512 -> Embedding is 1024 dimensional
            model_id = 'aari1995/German_Semantic_STS_V2'

            models_folder = os.environ.get('MODELS_FOLDER', '/opt/qa_service/models/')
            device = 'cpu'
            if torch.cuda.is_available():
                log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                mem_info = torch.cuda.mem_get_info(0)
                vram = round(mem_info[1] / 1024**3, 1)
                log.info(
                    f"VRAM available: {round(mem_info[0]/1024**3,1)} GB out of {vram} GB")
                if (vram >= 4):
                    device = 'cuda:0'
            log.info(f"Loading embedding model {model_id!r} in folder {models_folder!r}...")
            self.model = SentenceTransformer(model_id, device=device, cache_folder=models_folder)
            self.tokenizer = self.model.tokenizer
            log.info(f"Max Sequence Length: {self.max_sequence_length}")
            log.info(f"Embedding Dimensions: {self.embedding_dimensions}")
            log.info("...done.")
            if device != 'cpu':
                log.info(f"Embed 'Test': {self.embed(['Test'])}")  # Trigger CUDA loading
                log.info(f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")
            self._initialized = True

    @property
    def max_sequence_length(self) -> int:
        return self.model.get_max_seq_length()  # type: ignore

    @property
    def embedding_dimensions(self) -> int:
        return self.model.get_sentence_embedding_dimension()  # type: ignore

    def embed(self, paragraphs: list[str]) -> torch.Tensor:
        log.debug(f"Embedding {len(paragraphs)} paragraphs...")
        start = timer()
        embeddings = self.model.encode(paragraphs, convert_to_tensor=True)
        # L2-normalize -> dot-score is then same like cosine-similarity
        embeddings = embeddings / torch.sqrt((embeddings**2).sum(1, keepdims=True))  # type:ignore
        log.debug(f"...done in {timer() - start:.3f}s")
        assert isinstance(embeddings, torch.Tensor)
        return embeddings
