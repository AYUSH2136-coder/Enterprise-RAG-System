# src/generation/llm_client.py
# Responsibility: Load a local Qwen2.5 model and generate answers from prompts.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ---------- Lazy Loading ----------
# Model and tokenizer are loaded only once, on first call to generate_answer().
# This avoids crashing at import time if GPU/RAM is unavailable.
_tokenizer = None
_model = None


def _load_model():
    """Load tokenizer and model into memory. Called once on first use."""
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        print(f"[llm_client] Loading model: {MODEL_NAME} ...")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Set pad_token to eos_token to avoid padding warnings during generation
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,  # Half precision → saves ~50% VRAM
            device_map="auto"           # Auto-assigns to GPU if available, else CPU
        )

        print(f"[llm_client] Model loaded on device: {_model.device}")


def generate_answer(prompt: str, max_tokens: int = 1024, model: str = None) -> str:
    """
    model parameter accepted but unused — Qwen model is 
    loaded locally via MODEL_NAME constant at top of file.
    """
    _load_model()

    messages = [{"role": "user", "content": prompt}]

    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = _tokenizer(
        text,
        return_tensors="pt"
    ).to(_model.device)

    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.2,
        pad_token_id=_tokenizer.pad_token_id
    )

    response = _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response