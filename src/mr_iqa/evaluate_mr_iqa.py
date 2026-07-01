#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None
    Qwen2VLForConditionalGeneration = None


PROMPT = (
    "What is your overall rating on the quality of this picture? "
    "The rating should be a float between 1 and 5, rounded to two decimal places, "
    "with 1 representing very poor quality and 5 representing excellent quality. "
    "Please only output the final answer with one score in <answer> </answer> tags."
)
ANSWER_RE = re.compile(r"<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>", re.I | re.S)
NUMBER_RE = re.compile(r"(?<![\d.])([1-5](?:\.\d+)?)(?![\d.])")


def load_rows(path: str) -> list[dict[str, Any]]:
    path_obj = Path(path)
    with open(path_obj, "r", encoding="utf-8") as f:
        if path_obj.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("data", "records", "train_records", "test_records"):
        rows = data.get(key)
        if isinstance(rows, list):
            return rows
    raise ValueError(f"Unsupported data structure in {path}")


def resolve_image_path(row: dict[str, Any], image_root: str) -> str | None:
    candidates = [row.get("image"), row.get("image_path"), row.get("img_path"), row.get("source_image")]
    for image in candidates:
        if image is None:
            continue
        image_str = str(image).strip()
        paths = []
        if os.path.isabs(image_str):
            paths.append(image_str)
        paths.append(os.path.join(image_root, image_str))
        paths.append(os.path.join(image_root, os.path.basename(image_str)))
        for path in paths:
            if os.path.exists(path):
                return path
    return None


def gold_score(row: dict[str, Any]) -> Optional[float]:
    candidates = [
        row.get("normalized_score"),
        row.get("gt_score_norm"),
        row.get("score"),
        row.get("human_score"),
        row.get("mos"),
        row.get("rating"),
        row.get("quality_score"),
        (row.get("human_annotation") or {}).get("normalized_score")
        if isinstance(row.get("human_annotation"), dict)
        else None,
        (row.get("target") or {}).get("score") if isinstance(row.get("target"), dict) else None,
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            score = float(value)
        except Exception:
            continue
        if score > 5.0:
            score = 1.0 + (score - 1.0) * 4.0 / 99.0
        return max(1.0, min(5.0, score))
    return None


def parse_raw_score(text: str) -> Optional[float]:
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if m:
        return float(m.group(1))
    nums = NUMBER_RE.findall(text)
    if nums:
        return float(nums[-1])
    return None


def parse_score(text: str) -> Optional[float]:
    raw = parse_raw_score(text)
    if raw is None:
        return None
    return max(1.0, min(5.0, raw))


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def model_class(model_path: str):
    lower = model_path.lower()
    if "qwen2.5" in lower and Qwen2_5_VLForConditionalGeneration is not None:
        return Qwen2_5_VLForConditionalGeneration
    if "qwen2-vl" in lower and Qwen2VLForConditionalGeneration is not None:
        return Qwen2VLForConditionalGeneration
    return AutoModelForImageTextToText


def generate_one(model, processor, image_path: str, args) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
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


def corr(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 2 or len(set(ys)) < 2:
        return math.nan, math.nan
    return float(pearsonr(xs, ys).statistic), float(spearmanr(xs, ys).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--adapter_model", default="")
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()

    rows = load_rows(args.data_file)
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    cls = model_class(args.model_name_or_path)
    model = cls.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).eval()
    if args.adapter_model:
        if PeftModel is None:
            raise ImportError("peft is required for --adapter_model evaluation")
        model = PeftModel.from_pretrained(model, args.adapter_model).eval()

    results = []
    missing = 0
    for idx, row in tqdm(list(enumerate(rows)), desc="eval"):
        image_path = resolve_image_path(row, args.image_root)
        gold = gold_score(row)
        if not image_path or gold is None or not os.path.exists(image_path):
            missing += 1
            continue
        try:
            completion = generate_one(model, processor, image_path, args)
            raw_pred = parse_raw_score(completion)
            pred = parse_score(completion)
        except Exception as exc:
            completion = f"ERROR: {exc}"
            raw_pred = None
            pred = None
        results.append(
            {
                "index": idx,
                "image_path": image_path,
                "gold_score": gold,
                "raw_pred_score": raw_pred,
                "pred_score": pred,
                "completion": completion,
                "row": row,
            }
        )

    valid = [x for x in results if x["pred_score"] is not None]
    plcc, srcc = corr([x["gold_score"] for x in valid], [x["pred_score"] for x in valid])
    raw_out_of_range = sum(
        x["raw_pred_score"] is not None and not (1.0 <= x["raw_pred_score"] <= 5.0) for x in valid
    )
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_model": args.adapter_model or None,
        "num_total": len(results),
        "num_valid": len(valid),
        "num_missing_or_bad_gold": missing,
        "raw_out_of_range": raw_out_of_range,
        "plcc": plcc,
        "srcc": srcc,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
