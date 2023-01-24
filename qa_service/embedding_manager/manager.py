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
        if hasattr(self, 'model'):
            return
        with EmbeddingManager.lock:
            # Load model for Sentence Embedding

            # Max token length is 512 -> Embedding is 1024 dimensional
            # This model is really good, good with rivers, bad with mayor - 9/10
            model_id = 'Sahajtomar/German-semantic'  # Model Size is around 1.3 GB

            # Max token length is 350 -> Embedding is 768 dimensional
            # This model is really good, small and just 768-dim, even though scores all quite similar - 8/10
            model_id = 'PM-AI/bi-encoder_msmarco_bert-base_german'  # Model Size is around 0.4 GB

            # Max token length is 512 -> Embedding is 1024 dimensional
            # This model is quite good, but major river info sentence is missing - 7/10
            # model_id = 'aari1995/gBERT-large-sts-v2' # Model Size is around 1.25 GB

            # Max token length is 512 -> Embedding is 768 dimensional
            # This model is quite bad, focussing on very short sentences - 4/10
            # model_id = 'setu4993/LaBSE' # Model Size is around 1.77 GB

            # Max token length is 128 -> Embedding is 768 dimensional
            # This model is really bad, mixing up rivers and climate stuff - 3/10
            # model_id = 'symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli' # Model Size is around 1 GB

            # Max token length is 512 -> Embedding is 768 dimensional
            # Needs: pip install 'protobuf<=3.20.1' --force-reinstall
            # This model is really bad, very often no rivers at all - 2/10
            # model_id = 'T-Systems-onsite/german-roberta-sentence-transformer-v2' # Model Size is around 1 GB

            # Max token length is 512 -> Embedding is 768 dimensional
            # Needs: pip install 'protobuf<=3.20.1' --force-reinstall
            # This model is really bad, very often no rivers at all - 2/10
            # model_id = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer' # Model Size is around 1 GB

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
            self.model = SentenceTransformer(model_id, device=device, cache_folder=models_folder)
            log.info(f"Tokens: {self.get_max_seq_length()}")
            log.info(f"Dimensions: {self.get_dimensions()}")
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

    def get_dimensions(self) -> int:
        return self.model.get_sentence_embedding_dimension()  # type: ignore

    def get_max_seq_length(self) -> int:
        return self.model.get_max_seq_length()  # type: ignore
