import torch

from config import DEVICE, MAX_NEW_TOKENS, MAX_PIXELS, MODEL_NAME_OR_PATH, TORCH_DTYPE
from src.prompts import build_extraction_prompt


def get_torch_dtype():
    name = (TORCH_DTYPE or "bfloat16").lower()
    if name == "bfloat16": return torch.bfloat16
    if name == "float16": return torch.float16
    return torch.float32


def safe_open_image(path) -> "Image.Image":
    from PIL import Image
    path = Path(path) if isinstance(path, str) else path
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def load_model_and_processor(model_path=None):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    path = model_path or MODEL_NAME_OR_PATH
    torch_dtype = get_torch_dtype()
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch_dtype, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    return model, processor, torch_dtype


def run_inference(image, model, processor, device, torch_dtype, prompt_text=None, max_new_tokens=None, max_pixels=None) -> str:
    max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
    max_pixels = max_pixels or MAX_PIXELS
    if prompt_text is None:
        prompt_text = build_extraction_prompt()
    enc = processor(text=[prompt_text], images=[image], padding=True, truncation=False, return_tensors="pt", max_pixels=max_pixels)
    enc = {k: v.to(device) for k, v in enc.items()}
    if "pixel_values" in enc and torch_dtype in (torch.bfloat16, torch.float16):
        enc["pixel_values"] = enc["pixel_values"].to(dtype=torch_dtype)
    with torch.no_grad():
        out_ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
    input_len = enc["input_ids"].shape[1]
    return processor.tokenizer.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()
