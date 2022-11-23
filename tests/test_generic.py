'''
This file was created by ]init[ AG 2022.

Tests for WhisperManager.
'''
import logging

log = logging.getLogger(__name__)


def test_generic():
    log.debug("TEST")

def test_transformer():
    import os
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_folder = os.environ.get('MODEL_FOLDER', '/opt/qa_service/models/')
    checkpoint = "bigscience/mt0-xxl-mt"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=model_folder)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, cache_dir=model_folder)

    inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))
