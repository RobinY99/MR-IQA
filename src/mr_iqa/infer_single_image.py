#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mr_iqa.evaluate_mr_iqa import build_messages, load_image, model_class, parse_raw_score, parse_score, prompt_config


def generate_single_image(model, processor, image_path: str, args) -> str:
    messages = build_messages(image_path, args.prompt_mode)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[load_image(image_path)], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            use_cache=True,
        )
    completion_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MR-IQA inference on one image.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--adapter_model", default="")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--prompt_mode", choices=("non_thinking", "thinking"), default="non_thinking")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    image_path = str(Path(args.image_path).expanduser().resolve())
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if args.max_new_tokens is None:
        args.max_new_tokens = 256 if args.prompt_mode == "thinking" else 64

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    cls = model_class(args.model_name_or_path)
    model = cls.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).eval()
    if args.adapter_model:
        if PeftModel is None:
            raise ImportError("peft is required for --adapter_model inference")
        model = PeftModel.from_pretrained(model, args.adapter_model).eval()

    completion = generate_single_image(model, processor, image_path, args)
    system_prompt, user_prompt = prompt_config(args.prompt_mode)
    payload = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_model": args.adapter_model or None,
        "image_path": image_path,
        "prompt_mode": args.prompt_mode,
        "system_prompt": system_prompt,
        "prompt": user_prompt,
        "completion": completion,
        "raw_pred_score": parse_raw_score(completion),
        "pred_score": parse_score(completion),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
