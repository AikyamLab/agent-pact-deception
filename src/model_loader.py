from __future__ import annotations
import yaml
import torch
from pathlib import Path
from typing import Callable
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoProcessor


# ====================== PATH CONFIGURATION ======================
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"
# ====================== CONFIG LOADER ======================
def load_models_config(config_path: Path = CONFIG_PATH) -> dict:
    """Read and return the full models_config.yaml as a dict."""
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {config_path}: expected a mapping at top level.")
    return data

    
# ====================== HELPER FUNCTIONS ======================
def _build_bnb_config(quant_cfg: dict) -> BitsAndBytesConfig:
    """Build BitsAndBytesConfig from the quantization block in models_config.yaml."""
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )
def _finalize_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    """Apply shared tokenizer settings (left padding + pad token fallback)."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_dtype(cfg: dict):
    """Return the torch_dtype for from_pretrained.

    'auto' (or missing) is passed through as the string "auto" so the checkpoint
    loads in its native precision (important for MXFP4 models like gpt-oss).
    Otherwise map the name to a torch dtype, e.g. 'bfloat16' -> torch.bfloat16.
    """
    dtype = cfg.get("dtype", "auto")
    return "auto" if dtype == "auto" else getattr(torch, dtype)


def _common_model_kwargs(cfg: dict, *, trust_remote_code: bool) -> dict:
    """Shared kwargs for model.from_pretrained() — used by the bnb-capable families.

    Quantization is optional: the bitsandbytes config is built only when a
    `quantization` block is present. Models without one (or with it commented
    out) load in full/native precision instead of raising.
    """
    kwargs = {
        "device_map": "auto",
        "torch_dtype": _resolve_dtype(cfg),
        "attn_implementation": "sdpa",
        "trust_remote_code": trust_remote_code,
    }
    quant_cfg = cfg.get("quantization")
    if quant_cfg:
        kwargs["quantization_config"] = _build_bnb_config(quant_cfg)
    return kwargs
# ====================== FAMILY-SPECIFIC LOADERS ======================
# Add new model families here in the future. Examples of models are shown below.
def _load_qwen(cfg: dict):
    """Load any Qwen model (Qwen2, Qwen3, etc.)."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    tokenizer = _finalize_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        **_common_model_kwargs(cfg, trust_remote_code=True),
    )
    model.eval()
    return model, tokenizer
def _load_deepseek(cfg: dict):
    """Load DeepSeek models (DeepSeek-R1-Distill-Llama-70B, etc.)."""
    print("   → DeepSeek: using trust_remote_code=True (required for R1-Distill models)")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    tokenizer = _finalize_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        **_common_model_kwargs(cfg, trust_remote_code=True),
    )
    model.eval()
    return model, tokenizer
def _load_gemma(cfg: dict):
    """Load Gemma 3 or Gemma 4 models (text-only compatible)."""
    print("   → Gemma: using AutoProcessor + AutoModelForCausalLM")
    processor = AutoProcessor.from_pretrained(cfg["path"], trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = _finalize_tokenizer(processor.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        **_common_model_kwargs(cfg, trust_remote_code=True),
    )
    model.eval()
    return model, processor
def _load_gptoss(cfg: dict):
    """Load OpenAI gpt-oss (gpt-oss-20b / gpt-oss-120b).

    gpt-oss ships MXFP4-quantized, so it must load AS-IS: no bitsandbytes, no
    forced dtype conversion. We pass torch_dtype="auto" and let device_map place
    it on the visible GPU. attn_implementation is left at the model default
    (gpt-oss uses attention sinks, so we do not force sdpa).
    """
    print("   → gpt-oss: MXFP4-native, loading without bitsandbytes")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    tokenizer = _finalize_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype="auto",     # keep MXFP4; do NOT convert or quantize
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer
# ====================== REGISTER NEW FAMILIES HERE ======================
# To add a new model family (e.g. Llama, Mistral):
# 1. Create a new _load_xxx(cfg) function above
# 2. Add it to the dictionary below
_FAMILY_LOADERS: dict[str, Callable] = {
    "qwen": _load_qwen,
    "deepseek": _load_deepseek,
    "gemma": _load_gemma,
    "gptoss": _load_gptoss,
    # "mistral": _load_mistral,  # ← future example
}
# ====================== MAIN LOADER ======================
def load_model(
    model_key: str,
    config_path: Path = CONFIG_PATH,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    """
    Load a model and tokenizer by key from models_config.yaml.
    Works for HF backend only (your vLLM backend uses a different path).
    """
    config = load_models_config(config_path)
    all_models = config.get("models", {})
    if not isinstance(all_models, dict):
        raise ValueError("Expected 'models' to be a mapping in models_config.yaml")
    if model_key not in all_models:
        available = list(all_models.keys())
        raise KeyError(f"Model key '{model_key}' not found. Available models: {available}")
    cfg = all_models[model_key]
    family = cfg.get("family", "")
    if family not in _FAMILY_LOADERS:
        raise ValueError(
            f"Unknown model family '{family}' for model '{model_key}'. "
            f"Registered families: {list(_FAMILY_LOADERS.keys())}"
        )
    print(f"\n{'=' * 60}")
    print(f"Loading model : {model_key}")
    print(f"Family        : {family}")
    print(f"Path          : {cfg['path']}")
    print(f"Max new tokens: {cfg.get('max_new_tokens')}")
    print(f"Generation cfg: {cfg.get('generation', {})}")
    print(f"{'=' * 60}")
    model, tokenizer = _FAMILY_LOADERS[family](cfg)
    print(f"\n✅ Model loaded successfully on: {next(model.parameters()).device}")
    print(f"{'=' * 60}\n")
    return model, tokenizer, cfg