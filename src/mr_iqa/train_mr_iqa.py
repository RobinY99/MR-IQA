#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trl import TrlParser
from trl.trainer.grpo_config import GRPOConfig

from mr_iqa.qwen_vl_grpo_trainer import QwenVLGRPOTrainerDS, build_peft_config


NON_THINKING_SYSTEM_PROMPT = (
    "You are an image quality assessment assistant. Output only the final score in <answer> </answer> tags."
)
NON_THINKING_USER_PROMPT = (
    "What is your overall rating on the quality of this picture? "
    "The rating should be a float between 1 and 5, rounded to two decimal places, "
    "with 1 representing very poor quality and 5 representing excellent quality. "
    "Please only output the final answer with one score in <answer> </answer> tags."
)
THINKING_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <thinking> </thinking> and <answer> </answer> tags, respectively, i.e., "
    "<thinking> reasoning process here </thinking><answer> answer here </answer>"
)
THINKING_USER_PROMPT = (
    "What is your overall rating on the quality of this picture? "
    "The rating should be a float between 1 and 5, rounded to two decimal places, "
    "with 1 representing very poor quality and 5 representing excellent quality. "
    'Return the final answer in JSON format with the following keys: "rating": The score.'
)

SYSTEM_PROMPT = NON_THINKING_SYSTEM_PROMPT
USER_PROMPT = NON_THINKING_USER_PROMPT

ANSWER_RE = re.compile(
    r'<answer>\s*(?:\{\s*"?rating"?\s*:\s*)?([+-]?\d+(?:\.\d+)?)(?:\s*\})?\s*</answer>',
    re.I | re.S,
)
NON_THINKING_ANSWER_FORMAT_RE = re.compile(
    r"(?:<think>.*?</think>\s*)?<answer>\s*[+-]?\d+(?:\.\d+)?\s*</answer>",
    re.I | re.S,
)
THINKING_ANSWER_FORMAT_RE = re.compile(
    r'<thinking>[\s\S]+?</thinking>\s*<answer>\s*\{\s*"rating"\s*:\s*[+-]?\d+(?:\.\d+)?\s*\}\s*</answer>',
    re.I | re.S,
)
ANSWER_FORMAT_RE = NON_THINKING_ANSWER_FORMAT_RE
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def prompt_config(prompt_mode: str):
    mode = str(prompt_mode).strip().lower().replace("-", "_")
    if mode == "non_thinking":
        return {
            "mode": mode,
            "system_prompt": NON_THINKING_SYSTEM_PROMPT,
            "user_prompt": NON_THINKING_USER_PROMPT,
            "answer_format_re": NON_THINKING_ANSWER_FORMAT_RE,
            "solution_template": lambda score: f"<answer>{score:.2f}</answer>",
        }
    if mode == "thinking":
        return {
            "mode": mode,
            "system_prompt": THINKING_SYSTEM_PROMPT,
            "user_prompt": THINKING_USER_PROMPT,
            "answer_format_re": THINKING_ANSWER_FORMAT_RE,
            "solution_template": lambda score: f'<answer>{{"rating": {score:.2f}}}</answer>',
        }
    raise ValueError("--prompt_mode must be either 'non_thinking' or 'thinking'")


@dataclass
class MRIQATrainingArguments:
    data_file: str = field(default="")
    data_files: list[str] = field(default_factory=list)
    image_root: Optional[str] = field(default=None)
    max_samples: Optional[int] = field(default=None)
    dataset_seed: int = field(default=42)

    reward_funcs: str = field(default="margin,format")
    variance_mode: str = field(default="unit")
    prompt_mode: str = field(default="non_thinking")
    min_gt_std: float = field(default=1e-4)

    model_name_or_path: Optional[str] = field(default=None)
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)

    use_lora: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj")
    lora_bias: str = field(default="none")

    attn_implementation: str = field(default="sdpa")
    torch_dtype: str = field(default="bfloat16")
    trust_remote_code: bool = field(default=True)

    debug_generation: bool = field(default=False)
    debug_generation_every: int = field(default=1)
    debug_generation_samples: int = field(default=2)
    debug_prompt_chars: int = field(default=1000)


class MRIQADataset:
    def __init__(self, args: MRIQATrainingArguments):
        self.args = args
        self.data_files = self._resolve_data_files(args)
        self.samples = self._load_samples(args)

    @staticmethod
    def _resolve_data_files(args):
        files = []
        if args.data_files:
            files.extend(args.data_files)
        if args.data_file:
            files.append(args.data_file)
        files = [x for x in files if x]
        if not files:
            raise ValueError("Please provide --data_files or --data_file")
        missing = [x for x in files if not os.path.exists(x)]
        if missing:
            raise ValueError(f"Missing data files: {missing}")
        return files

    @staticmethod
    def _load_rows(path):
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        for key in ("data", "records", "train_records", "test_records"):
            rows = data.get(key)
            if isinstance(rows, list):
                return rows
        raise ValueError(f"Unsupported data structure in {path}")

    def _resolve_image_path(self, image_path: str) -> str:
        image_path = str(image_path).strip()
        candidates = [image_path]
        if self.args.image_root:
            candidates.append(os.path.join(self.args.image_root, image_path))
            candidates.append(os.path.join(self.args.image_root, os.path.basename(image_path)))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[-1]

    @staticmethod
    def _extract_score(row: dict) -> Optional[float]:
        for key in (
            "score_norm",
            "source_score",
            "gt_score_norm",
            "normalized_score",
            "score",
            "human_score",
            "mos",
            "rating",
            "quality_score",
        ):
            if row.get(key) is None:
                continue
            try:
                return max(1.0, min(5.0, float(row[key])))
            except Exception:
                continue
        convs = row.get("conversations") or []
        if len(convs) >= 2:
            return extract_first_number(str(convs[1].get("value", "")), fallback=None)
        return None

    @staticmethod
    def _extract_std(row: dict) -> Optional[float]:
        for key in ("std_norm", "source_std", "std", "score_std", "mos_std"):
            if row.get(key) is None:
                continue
            try:
                std = float(row[key])
            except Exception:
                continue
            if std >= 0:
                return std
        return None

    def _load_samples(self, args):
        samples = []
        cfg = prompt_config(args.prompt_mode)
        for data_file in self.data_files:
            for row in self._load_rows(data_file):
                image_path = row.get("image") or row.get("image_path") or row.get("img_path")
                if not image_path:
                    continue
                image_path = self._resolve_image_path(str(image_path))
                if not os.path.exists(image_path):
                    continue
                score = self._extract_score(row)
                std = self._extract_std(row)
                if score is None or std is None:
                    continue
                samples.append(
                    {
                        "sample_id": row.get("id") or f"sample_{len(samples):07d}",
                        "image_path": image_path,
                        "solution": cfg["solution_template"](score),
                        "target_mean": float(score),
                        "target_std": max(float(std), float(args.min_gt_std)),
                        "system_prompt": cfg["system_prompt"],
                        "custom_question": cfg["user_prompt"],
                        "dataset_name": row.get("dataset_name", "iqa"),
                    }
                )
        if not samples:
            raise ValueError("No valid MR-IQA samples found.")
        seed = getattr(args, "_effective_data_seed", None)
        if seed is None:
            seed = args.dataset_seed
        rng = random.Random(seed)
        rng.shuffle(samples)
        if args.max_samples is not None:
            samples = samples[: int(args.max_samples)]
        print(f"Loaded {len(samples)} MR-IQA samples from {len(self.data_files)} file(s).", flush=True)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def completion_to_text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"]
    return str(completion)


def clamp_score(value: float) -> float:
    if not math.isfinite(value):
        return random.uniform(1, 5)
    return max(1.0, min(5.0, float(value)))


def extract_first_number(text: str, fallback=random.uniform):
    m = ANSWER_RE.search(text or "")
    target = m.group(1) if m else text or ""
    nums = NUMBER_RE.findall(target)
    if nums:
        return clamp_score(float(nums[-1]))
    if fallback is None:
        return None
    return clamp_score(float(fallback(1, 5)))


def extract_answer_number(text: str) -> Optional[float]:
    m = ANSWER_RE.search(text or "")
    if not m:
        return None
    try:
        value = float(m.group(1))
    except Exception:
        return None
    if not math.isfinite(value) or value < 1.0 or value > 5.0:
        return None
    return value


MARGIN_VARIANCE_MODE = "unit"
MARGIN_CALL_COUNT = 0


def margin_scale(gt_std_i=None, gt_std_j=None) -> float:
    if MARGIN_VARIANCE_MODE == "unit":
        return 1.0
    if MARGIN_VARIANCE_MODE == "sigma":
        left = max(float(gt_std_i or 0.0), 0.0)
        right = max(float(gt_std_j or 0.0), 0.0)
        return max(math.sqrt(left * left + right * right), 1e-4)
    raise ValueError("--variance_mode must be either 'unit' or 'sigma'")


def margin_reward_value(pred_delta, gt_delta, gt_std_i=None, gt_std_j=None):
    signed_error = float(pred_delta) - float(gt_delta)
    normalized_error = signed_error / margin_scale(gt_std_i, gt_std_j)
    squared_error = normalized_error * normalized_error
    return math.exp(-0.5 * squared_error), signed_error, squared_error


def group_by_sample_id(indices, sample_ids, target_mean, target_std):
    groups = []
    current = []
    current_key = object()
    for idx in indices:
        if sample_ids is not None and idx < len(sample_ids):
            key = sample_ids[idx]
        else:
            key = (target_mean[idx] if idx < len(target_mean) else None, target_std[idx] if idx < len(target_std) else None)
        if current and key != current_key:
            groups.append((current_key, current))
            current = []
        current_key = key
        current.append(idx)
    if current:
        groups.append((current_key, current))
    return groups


def margin_reward(completions, solution=None, target_mean=None, target_std=None, **kwargs):
    if target_mean is None:
        target_mean = [extract_first_number(str(s), fallback=None) for s in (solution or [])]
    if target_std is None:
        target_std = [1.0 for _ in completions]

    sample_ids = kwargs.get("sample_id") or kwargs.get("sample_ids")
    preds = [extract_answer_number(completion_to_text(c)) for c in completions]
    rewards = [0.0 for _ in completions]
    groups = group_by_sample_id(list(range(len(completions))), sample_ids, target_mean, target_std)

    group_stats = []
    for _, idxs in groups:
        vals = [float(preds[i]) for i in idxs if preds[i] is not None]
        mean_pred = sum(vals) / len(vals) if vals else 0.0
        label = target_mean[idxs[0]] if idxs and idxs[0] < len(target_mean) else None
        std = target_std[idxs[0]] if idxs and idxs[0] < len(target_std) else 1.0
        group_stats.append((idxs, vals, mean_pred, label, max(float(std), 1e-4)))

    margin_values = []
    error_values = []
    l2_values = []
    pair_counts = []

    for gi, (idxs, _vals, _mean_pred, label, gt_std) in enumerate(group_stats):
        for idx in idxs:
            pred = preds[idx]
            if pred is None or idx >= len(target_mean) or target_mean[idx] is None:
                rewards[idx] = 0.0
                continue
            try:
                pair_sum = 0.0
                pair_count = 0
                for gj, (_other_idxs, other_vals, other_mean_pred, other_label, other_gt_std) in enumerate(group_stats):
                    if gj == gi or other_label is None or label is None or not other_vals:
                        continue
                    gt_delta = float(label) - float(other_label)
                    pred_delta = float(pred) - float(other_mean_pred)
                    reward, signed_error, l2_error = margin_reward_value(pred_delta, gt_delta, gt_std, other_gt_std)
                    pair_sum += reward
                    margin_values.append(reward)
                    error_values.append(abs(signed_error))
                    l2_values.append(l2_error)
                    pair_count += 1
                if pair_count:
                    pair_counts.append(pair_count)
                rewards[idx] = float(pair_sum / pair_count) if pair_count else 0.0
            except Exception:
                rewards[idx] = 0.0
    log_margin_metrics(rewards, margin_values, error_values, pair_counts, preds, group_stats, l2_values)
    return rewards


def _mean_std(vals):
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, math.sqrt(var)


def _quantile(vals, q):
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _sigma_payload(prefix, vals):
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    mean, std = _mean_std(vals)
    return {
        f"{prefix}_mean": round(mean, 6),
        f"{prefix}_std": round(std, 6),
        f"{prefix}_min": round(min(vals), 6) if vals else 0.0,
        f"{prefix}_p10": round(_quantile(vals, 0.10), 6),
        f"{prefix}_p50": round(_quantile(vals, 0.50), 6),
        f"{prefix}_p90": round(_quantile(vals, 0.90), 6),
        f"{prefix}_max": round(max(vals), 6) if vals else 0.0,
    }


def log_margin_metrics(rewards, margin_values, error_values, pair_counts, preds, group_stats, l2_values=None):
    global MARGIN_CALL_COUNT
    MARGIN_CALL_COUNT += 1
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    if str(rank) != "0" or MARGIN_CALL_COUNT % 50 != 0:
        return
    r_mean, r_std = _mean_std(rewards)
    m_mean, m_std = _mean_std(margin_values)
    e_mean, e_std = _mean_std(error_values)
    l2_mean, l2_std = _mean_std(l2_values or [])
    parsed = sum(1 for p in preds if p is not None)
    unique = len({round(float(p), 4) for p in preds if p is not None})
    total = len(preds) if preds else 1

    pred_sigma_sample = []
    gt_sigma = []
    sigma_ratio_sample_to_gt = []
    valid_generations = []
    for _idxs, vals, _mean_pred, _label, cur_gt_std in group_stats:
        vals = [float(v) for v in vals if math.isfinite(float(v))]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        sample_var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
        sample_sigma = math.sqrt(max(sample_var, 0.0))
        cur_gt_std = max(float(cur_gt_std), 1e-4)
        pred_sigma_sample.append(sample_sigma)
        gt_sigma.append(cur_gt_std)
        sigma_ratio_sample_to_gt.append(sample_sigma / cur_gt_std)
        valid_generations.append(len(vals))

    payload = {
        "margin_metrics": True,
        "reward_call": MARGIN_CALL_COUNT,
        "margin_reward_mean": round(r_mean, 6),
        "margin_reward_std": round(r_std, 6),
        "pair_margin_mean": round(m_mean, 6),
        "pair_margin_std": round(m_std, 6),
        "raw_margin_error_mean": round(e_mean, 6),
        "raw_margin_error_std": round(e_std, 6),
        "raw_margin_l2_error_mean": round(l2_mean, 6),
        "raw_margin_l2_error_std": round(l2_std, 6),
        "variance_mode": MARGIN_VARIANCE_MODE,
        "avg_pair_count": round(sum(pair_counts) / len(pair_counts), 6) if pair_counts else 0.0,
        "parse_success_rate": round(parsed / total, 6),
        "unique_score_ratio": round(unique / total, 6),
        "avg_valid_generations_per_sample": round(sum(valid_generations) / len(valid_generations), 6)
        if valid_generations
        else 0.0,
        "reward_name": "margin",
        "timestamp": time.strftime("%F %T %Z"),
    }
    payload.update(_sigma_payload("pred_sigma_sample", pred_sigma_sample))
    payload.update(_sigma_payload("gt_sigma", gt_sigma))
    payload.update(_sigma_payload("sigma_ratio_sample_to_gt", sigma_ratio_sample_to_gt))
    line = json.dumps(payload, ensure_ascii=False)
    log_path = os.environ.get("VARIANCE_LOG_PATH", "")
    if log_path:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            print(json.dumps({"variance_log_error": str(exc), "path": log_path}, ensure_ascii=False), flush=True)
    print(line, flush=True)


margin_reward.__name__ = "margin_reward"


def format_reward(completions, **kwargs):
    return [1.0 if ANSWER_FORMAT_RE.fullmatch(completion_to_text(c).strip()) else 0.0 for c in completions]


format_reward.__name__ = "format_reward"


def build_reward_funcs(spec: str):
    registry = {"margin": margin_reward, "format": format_reward}
    funcs = []
    for name in [x.strip() for x in spec.split(",") if x.strip()]:
        funcs.append(registry[name])
    return funcs


def main():
    parser = TrlParser((MRIQATrainingArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_and_config()
    script_args._effective_data_seed = getattr(training_args, "data_seed", None)

    global MARGIN_VARIANCE_MODE, ANSWER_FORMAT_RE
    MARGIN_VARIANCE_MODE = str(script_args.variance_mode).strip().lower()
    if MARGIN_VARIANCE_MODE not in {"unit", "sigma"}:
        raise ValueError("--variance_mode must be either 'unit' or 'sigma'")
    prompt_cfg = prompt_config(script_args.prompt_mode)
    script_args.prompt_mode = prompt_cfg["mode"]
    ANSWER_FORMAT_RE = prompt_cfg["answer_format_re"]

    if script_args.model_name_or_path:
        training_args.model_name_or_path = script_args.model_name_or_path
    init_kwargs = dict(getattr(training_args, "model_init_kwargs", {}) or {})
    init_kwargs["trust_remote_code"] = script_args.trust_remote_code
    training_args.model_init_kwargs = init_kwargs

    dataset = MRIQADataset(script_args)
    trainer = QwenVLGRPOTrainerDS(
        model=training_args.model_name_or_path,
        reward_funcs=build_reward_funcs(script_args.reward_funcs),
        args=training_args,
        train_dataset=dataset,
        peft_config=build_peft_config(script_args),
        eval_dataset=None,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        attn_implementation=script_args.attn_implementation,
        torch_dtype=script_args.torch_dtype,
        trust_remote_code=script_args.trust_remote_code,
        debug_generation=script_args.debug_generation,
        debug_generation_every=script_args.debug_generation_every,
        debug_generation_samples=script_args.debug_generation_samples,
        debug_prompt_chars=script_args.debug_prompt_chars,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    if trainer.is_world_process_zero():
        trainer.processing_class.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
