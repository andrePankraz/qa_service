"""
This file was created by ]init[ AG 2023.

Module for Text Generation Models.
"""
import os
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
        if not hasattr(cls, "instance"):
            cls.instance = super(T2tManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, "pipeline"):
            return
        with T2tManager.lock:
            # Load model for Text-to-Text Generation

            ##### text2text-generation ####

            # T5 was trained for max sequence length 512, but can handle longer sequences!
            # There will be a warning and the model might not attend to all facts, but it works.
            # Token indices sequence length is longer than the specified maximum
            # sequence length for this model (683 > 512). Running this sequence
            # through the model will result in indexing errors
            # model_id = 'google/flan-t5-xl'  # Model Size is around 10.6 GB

            # Needs at least 24 GB VRAM?!
            # model_id = 'google/flan-t5-xxl'  # Model Size is around 41.9 GB

            # Needs at least 20 GB VRAM?!
            # model_id = 'google/flan-ul2'  # Model Size is around 36.7 GB

            # Needs: pip install protobuf==3.20.*
            # Tokenization seems defect & Isn't instruction tuned
            # model_id = 'google/mt5-xl'  # Model Size is around 13.9 GB

            ##### text-generation ####
            # Decoder-only - small max sequence length is very punishing!
            # max_new_tokens=512

            # Replace in snapshots/5f.../tokenizer_config.json "tokenizer_class": "LLaMATokenizer" with LlamaTokenizer
            # model_id = 'decapoda-research/llama-7b-hf'  # Model Size is around 12.5 GB
            # model_id = 'decapoda-research/llama-13b-hf'  # Model Size is around 36.3 GB

            # Model Size is around 26 GB, Multilanguage, Instruction tuned,
            # Max sequence length is 8k
            model_id = "Salesforce/xgen-7b-8k-inst"

            models_folder = os.environ.get("MODELS_FOLDER", "/opt/qa_service/models/")

            device = "cpu"
            if torch.cuda.is_available():
                log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                mem_info = torch.cuda.mem_get_info(0)
                vram = round(mem_info[1] / 1024**3, 1)
                log.info(f"VRAM available: {round(mem_info[0]/1024**3,1)} GB out of {vram} GB")
                if vram >= 4:
                    device = "cuda:0"
            log.info(f"Loading model {model_id!r} in folder {models_folder!r}...")
            self.pipeline = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=model_id,
                device=device,
                trust_remote_code=True,
                model_kwargs={
                    "cache_dir": models_folder,
                    "torch_dtype": torch.bfloat16,
                    # 'device_map': 'auto',
                    # 'load_in_8bit': True
                },
            )
            log.info(f"Max Sequence Length: {self.get_max_sequence_length()}")
            log.info("...done.")
            if device != "cpu":
                log.info(f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")

    def get_max_sequence_length(self) -> int:
        return self.pipeline.tokenizer.model_max_length  # type: ignore

    def generate(self, prompt: str) -> str:
        log.debug(f"Generating...")
        start = timer()
        response: list[dict] = self.pipeline(prompt, max_new_tokens=200)  # type: ignore
        log.debug(f"...done in {timer() - start:.3f}s")
        # assert isinstance(answers, list)
        return response[0]["generated_text"]

    def answer(self, question: str, context: str) -> str:
        log.debug(f"Answering...")
        start = timer()

        prompt = f"""\
Du bist ein hilfreicher, ehrlicher und harmloser KI-Assistent.
Du beantwortest Fragen, basierend auf gegebenen Fakten.

Nutze nur folgende Fakten zur Beantwortung der nachfolgend gestellten Frage:
<BEGIN FAKTEN>
{context}
</END FAKTEN>

Frage: {question}

Antworte in einem kurzen und prägnanten Satz:"""

        log.debug(f"Prompt:\n{prompt}")

        response: list[dict] = self.pipeline(prompt, max_new_tokens=200)  # type: ignore
        log.debug(f"...done in {timer() - start:.3f}s")
        return response[0]["generated_text"]
