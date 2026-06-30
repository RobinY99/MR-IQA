#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Callable, Optional, Union, Sized

import requests
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Sampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None
    Qwen2VLForConditionalGeneration = None

from datasets import Dataset, IterableDataset
from packaging import version
from trl.trainer.grpo_config import GRPOConfig
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.utils import is_peft_model, set_seed

if is_peft_available():
    from peft import LoraConfig, PeftConfig, get_peft_model
else:
    LoraConfig = None
    PeftConfig = None
    get_peft_model = None

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = 196608):
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        bg = Image.new("RGB", pil_image.size, (255, 255, 255))
        bg.paste(pil_image, mask=pil_image.split()[3])
        return bg
    return pil_image.convert("RGB")


def fetch_image(ele: dict[str, Any]) -> Image.Image:
    image = ele.get("image") or ele.get("image_url")
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str) and image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = Image.open(bio).copy()
    elif isinstance(image, str):
        image_obj = Image.open(image)
    else:
        raise ValueError(f"Unsupported image input: {type(image)}")
    image_obj = to_rgb(image_obj)
    width, height = image_obj.size
    max_pixels = ele.get("max_pixels", 196608)
    min_pixels = ele.get("min_pixels", MIN_PIXELS)
    resized_height, resized_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    return image_obj.resize((resized_width, resized_height))


def process_vision_info(conversations: list[list[dict]]) -> tuple[list[Image.Image] | None, None]:
    image_inputs = []
    for conversation in conversations:
        for message in conversation:
            content = message.get("content", [])
            if isinstance(content, list):
                for ele in content:
                    if ele.get("type") in ("image", "image_url") or "image" in ele or "image_url" in ele:
                        image_inputs.append(fetch_image(ele))
    return (image_inputs if image_inputs else None), None


class RepeatRandomSampler(Sampler):
    def __init__(self, data_source: Sized, mini_repeat_count: int, batch_size: int = 1, repeat_count: int = 1, seed: Optional[int] = None):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        chunks = [indexes[i:i+self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        chunks = [c for c in chunks if len(c) == self.batch_size]
        for chunk in chunks:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self):
        return self.num_samples * self.mini_repeat_count * self.repeat_count


def _as_torch_dtype(x):
    if isinstance(x, torch.dtype): return x
    if x in (None, "auto"): return "auto"
    return getattr(torch, str(x))


def load_vl_model_and_processor(model_name_or_path: str, model_init_kwargs: dict, max_pixels: int, min_pixels: int):
    name = str(model_name_or_path)
    lower = name.lower()
    kwargs = dict(model_init_kwargs)
    kwargs.setdefault("trust_remote_code", True)

    if "torch_dtype" in kwargs:
        kwargs["torch_dtype"] = _as_torch_dtype(kwargs["torch_dtype"])

    kwargs.pop("use_cache", None)

    trust_remote_code = kwargs.get("trust_remote_code", True)

    cfg = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
    cfg_cls = cfg.__class__.__name__.lower()
    model_type = str(getattr(cfg, "model_type", "")).lower()

    is_qwen3_vl = (
        "qwen3-vl" in lower
        or "qwen3_vl" in lower
        or "qwen3vl" in cfg_cls
        or "qwen3_vl" in model_type
        or "qwen3-vl" in model_type
    )
    is_qwen25_vl = (
        "qwen2.5-vl" in lower
        or "qwen2_5_vl" in lower
        or "qwen2_5vl" in cfg_cls
        or "qwen2_5_vl" in model_type
    )
    is_qwen2_vl = (
        "qwen2-vl" in lower
        or "qwen2_vl" in lower
        or "qwen2vl" in cfg_cls
        or "qwen2_vl" in model_type
    )

    if is_qwen3_vl:
        try:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(name, **kwargs)
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=trust_remote_code)

    elif is_qwen25_vl:
        if Qwen2_5_VLForConditionalGeneration is not None:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **kwargs)
        else:
            model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=trust_remote_code)

    elif is_qwen2_vl:
        if Qwen2VLForConditionalGeneration is not None:
            model = Qwen2VLForConditionalGeneration.from_pretrained(name, **kwargs)
        else:
            model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=trust_remote_code)

    else:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        processor = AutoTokenizer.from_pretrained(
            name,
            padding_side="left",
            trust_remote_code=trust_remote_code,
        )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        pad_token_id = processor.tokenizer.pad_token_id
        eos_token_id = processor.tokenizer.eos_token_id
        processor.pad_token_id = pad_token_id
        processor.eos_token_id = eos_token_id
    else:
        processor.padding_side = "left"
        pad_token_id = processor.pad_token_id
        eos_token_id = processor.eos_token_id

    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.pad_token_id = eos_token_id
        processor.pad_token_id = eos_token_id

    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min_pixels

    print(
        f"[load_vl_model] name={name}, config={cfg.__class__.__name__}, "
        f"model_type={model_type}, qwen3_vl={is_qwen3_vl}, qwen25_vl={is_qwen25_vl}, qwen2_vl={is_qwen2_vl}"
    )
    return model, processor, pad_token_id, eos_token_id


class QwenVLGRPOTrainerDS(Trainer):
    def __init__(self, model: Union[str, PreTrainedModel], reward_funcs: Union[RewardFunc, list[RewardFunc]], args: GRPOConfig = None,
                 train_dataset: Optional[Union[Dataset, IterableDataset]] = None, eval_dataset=None,
                 processing_class: Optional[PreTrainedTokenizerBase] = None, reward_processing_classes=None,
                 callbacks: Optional[list[TrainerCallback]] = None, optimizers=(None, None), peft_config: Optional["PeftConfig"] = None,
                 max_pixels: int = 196608, min_pixels: int = 3136, attn_implementation: str = "sdpa", torch_dtype: str = "bfloat16", trust_remote_code: bool = True,
                 debug_generation: bool = False, debug_generation_every: int = 1, debug_generation_samples: int = 2, debug_prompt_chars: int = 1500):
        if args is None:
            name = model if isinstance(model, str) else model.config._name_or_path
            args = GRPOConfig(f"{str(name).split('/')[-1]}-GRPO")

        model_init_kwargs = dict(args.model_init_kwargs or {})
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs.setdefault("torch_dtype", torch_dtype)
        model_init_kwargs.setdefault("trust_remote_code", trust_remote_code)
        model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache", False)

        if isinstance(model, str):
            model_id = model
            model, loaded_processor, pad_token_id, eos_token_id = load_vl_model_and_processor(model_id, model_init_kwargs, max_pixels, min_pixels)
        else:
            model_id = model.config._name_or_path
            loaded_processor = processing_class
            if processing_class is None:
                loaded_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            if hasattr(loaded_processor, "tokenizer"):
                pad_token_id = loaded_processor.tokenizer.pad_token_id
                eos_token_id = loaded_processor.tokenizer.eos_token_id
            else:
                pad_token_id = loaded_processor.pad_token_id
                eos_token_id = loaded_processor.eos_token_id

        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        if peft_config is not None:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model, _, _, _ = load_vl_model_and_processor(model_id, model_init_kwargs, max_pixels, min_pixels)
        else:
            self.ref_model = create_reference_model(model)

        processing_class = processing_class or loaded_processor
        self.processing_class = processing_class

        if not isinstance(reward_funcs, list): reward_funcs = [reward_funcs]
        for i, rf in enumerate(reward_funcs):
            if isinstance(rf, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(rf, num_labels=1, **model_init_kwargs)
        self.reward_funcs = reward_funcs

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features): return features

        self.max_prompt_length = getattr(args, "max_prompt_length", None)
        self.max_completion_length = getattr(args, "max_completion_length", 512)
        self.num_generations = getattr(args, "num_generations", 2)
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=getattr(args, "temperature", 1.0),
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        self.beta = getattr(args, "beta", 0.0)
        self.epsilon = getattr(args, "epsilon", getattr(args, "epsilon_low", 0.2))
        self.num_iterations = getattr(args, "num_iterations", 1)
        self._step = 0
        self._buffered_inputs = [None] * args.gradient_accumulation_steps
        self._metrics = defaultdict(list)

        self.debug_generation = bool(debug_generation)
        self.debug_generation_every = max(1, int(debug_generation_every))
        self.debug_generation_samples = max(1, int(debug_generation_samples))
        self.debug_prompt_chars = max(200, int(debug_prompt_chars))

        super().__init__(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=eval_dataset, processing_class=processing_class, callbacks=callbacks, optimizers=optimizers)

        global_batch = args.per_device_train_batch_size * self.accelerator.num_processes
        possible = [n for n in range(2, global_batch + 1) if global_batch % n == 0]
        if self.num_generations not in possible:
            raise ValueError(f"global train batch size {global_batch} must be divisible by num_generations {self.num_generations}; valid={possible}")
        set_seed(args.seed, device_specific=True)
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model, args):
        model.config.use_cache = False
        if is_peft_model(model): model.base_model.gradient_checkpointing_enable()
        else: model.gradient_checkpointing_enable()
        kwargs = args.gradient_checkpointing_kwargs or {}
        if kwargs.get("use_reentrant", True): model.enable_input_require_grads()
        return model

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_tokenizer_for_mmids(self):
        if hasattr(self.processing_class, "tokenizer"):
            return self.processing_class.tokenizer
        return self.processing_class

    def _resolve_mm_token_ids(self):
        tok = self._get_tokenizer_for_mmids()
        try:
            cfg = getattr(self.accelerator.unwrap_model(self.model), "config", None)
        except Exception:
            cfg = getattr(self.model, "config", None)

        ids = {}
        for name in ["image_token_id", "video_token_id"]:
            val = getattr(cfg, name, None) if cfg is not None else None
            if isinstance(val, int):
                ids[name] = val

        candidates = {
            "image_token_id": ["<|image_pad|>", "<image>", "<|image|>"],
            "video_token_id": ["<|video_pad|>", "<video>", "<|video|>"],
        }
        for key, toks in candidates.items():
            if key in ids:
                continue
            for t in toks:
                try:
                    tid = tok.convert_tokens_to_ids(t)
                    if isinstance(tid, int) and tid >= 0 and tid != getattr(tok, "unk_token_id", None):
                        ids[key] = tid
                        break
                except Exception:
                    pass
        return ids

    def _build_mm_token_type_ids_from_input_ids(self, input_ids):
        mm_token_type_ids = torch.zeros_like(input_ids)
        ids = self._resolve_mm_token_ids()
        image_id = ids.get("image_token_id", None)
        video_id = ids.get("video_token_id", None)

        if image_id is not None:
            mm_token_type_ids = torch.where(
                input_ids == image_id,
                torch.ones_like(mm_token_type_ids),
                mm_token_type_ids,
            )
        if video_id is not None:
            mm_token_type_ids = torch.where(
                input_ids == video_id,
                torch.full_like(mm_token_type_ids, 2),
                mm_token_type_ids,
            )
        return mm_token_type_ids


    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
    ):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw
        if mm_token_type_ids is None and image_grid_thw is not None:
            mm_token_type_ids = self._build_mm_token_type_ids_from_input_ids(input_ids)
        if mm_token_type_ids is not None:
            kwargs["mm_token_type_ids"] = mm_token_type_ids
        logits = model(**kwargs).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        outs = []
        for row_logits, row_ids in zip(logits, labels):
            log_probs = row_logits.log_softmax(dim=-1)
            outs.append(torch.gather(log_probs, dim=1, index=row_ids.unsqueeze(1)).squeeze(1))
        return torch.stack(outs)

    def _prepare_inputs(self, inputs): return inputs

    def _generate_and_score_completions(self, inputs, model):
        device = self.accelerator.device
        raw_messages, text_list = [], []
        for ex in inputs:
            msg = [
                {"role": "system", "content": [{"type": "text", "text": ex["system_prompt"]}]},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{ex['image_path']}", "max_pixels": getattr(self.processing_class.image_processor, "max_pixels", 196608), "min_pixels": getattr(self.processing_class.image_processor, "min_pixels", 3136)},
                    {"type": "text", "text": ex["custom_question"]},
                ]},
            ]
            raw_messages.append(msg)
            text_list.append(self.processing_class.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

        images, videos = process_vision_info(raw_messages)
        processor_kwargs = dict(text=text_list, images=images, return_tensors="pt", padding=True)
        if videos is not None: processor_kwargs["videos"] = videos
        prompt_inputs = self.processing_class(**processor_kwargs)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs.get("pixel_values", None)
        image_grid_thw = prompt_inputs.get("image_grid_thw", None)
        mm_token_type_ids = prompt_inputs.get("mm_token_type_ids", None)
        if mm_token_type_ids is None and image_grid_thw is not None:
            mm_token_type_ids = self._build_mm_token_type_ids_from_input_ids(prompt_ids)

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            prompt_completion_ids = unwrapped.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        if mm_token_type_ids is not None:
            completion_mm_token_type_ids = torch.zeros(
                (mm_token_type_ids.size(0), completion_ids.size(1)),
                dtype=mm_token_type_ids.dtype,
                device=mm_token_type_ids.device,
            )
            full_mm_token_type_ids = torch.cat([mm_token_type_ids, completion_mm_token_type_ids], dim=1)
        else:
            full_mm_token_type_ids = None

        eos_id = self.processing_class.eos_token_id if hasattr(self.processing_class, "eos_token_id") else self.processing_class.tokenizer.eos_token_id
        is_eos = completion_ids == eos_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        with torch.no_grad():
            if self.num_iterations > 1:
                old_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw, full_mm_token_type_ids)[:, prompt_length - 1:]
            else:
                old_logps = None
            if self.beta == 0.0:
                ref_logps = None
            elif self.ref_model is not None:
                ref_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw, full_mm_token_type_ids)[:, prompt_length - 1:]
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw, full_mm_token_type_ids)[:, prompt_length - 1:]

        tokenizer = self.processing_class.tokenizer if hasattr(self.processing_class, "tokenizer") else self.processing_class
        decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": x}] for x in decoded]
        prompts = text_list

        if self.debug_generation and self.accelerator.is_main_process and self.state.global_step % self.debug_generation_every == 0:
            print("\n" + "=" * 100, flush=True)
            print(f"[DEBUG GENERATION] global_step={self.state.global_step}", flush=True)
            show_n = min(self.debug_generation_samples, len(completions))
            for dbg_i in range(show_n):
                print("-" * 100, flush=True)
                print(f"[Sample {dbg_i}] image_path={inputs[dbg_i].get('image_path', '')}", flush=True)
                print("[Gold score]", inputs[dbg_i].get("solution", None), flush=True)
                print("[Prompt tail]", flush=True)
                print(prompts[dbg_i][-self.debug_prompt_chars:], flush=True)
                print("[Completion]", flush=True)
                print(completions[dbg_i][0]["content"], flush=True)
            print("=" * 100 + "\n", flush=True)

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (rf, rpc) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(rf, PreTrainedModel):
                texts = [p + c[0]["content"] for p, c in zip(prompts, completions)]
                reward_inputs = rpc(texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode(): rewards_per_func[:, i] = rf(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {k: [ex[k] for ex in inputs] for k in inputs[0].keys() if k not in ["prompt", "completion"]}
                vals = rf(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(vals, dtype=torch.float32, device=device)

        rewards_per_func_all = self.accelerator.gather(rewards_per_func)
        rewards = rewards_per_func_all.sum(dim=1)

        if self.debug_generation and self.accelerator.is_main_process and self.state.global_step % self.debug_generation_every == 0:
            print("\n" + "=" * 100, flush=True)
            print(f"[DEBUG REWARD] global_step={self.state.global_step}", flush=True)
            print("[Reward funcs]", [getattr(rf, "__name__", str(rf)) for rf in self.reward_funcs], flush=True)
            print("[rewards_per_func local shape]", tuple(rewards_per_func.shape), flush=True)
            print(rewards_per_func[: min(8, rewards_per_func.size(0))].detach().float().cpu(), flush=True)
            print("[rewards gathered first rows]", flush=True)
            print(rewards[: min(16, rewards.size(0))].detach().float().cpu(), flush=True)
            print("=" * 100 + "\n", flush=True)

        mean_grouped = rewards.view(-1, self.num_generations).mean(dim=1).repeat_interleave(self.num_generations)
        std_grouped = rewards.view(-1, self.num_generations).std(dim=1).repeat_interleave(self.num_generations)
        advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)
        sl = slice(self.accelerator.process_index * len(prompts), (self.accelerator.process_index + 1) * len(prompts))
        advantages = advantages[sl]

        self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item())
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func_all).mean(0)
        for i, rf in enumerate(self.reward_funcs):
            name = rf.config._name_or_path.split("/")[-1] if hasattr(rf, "config") else rf.__name__
            self._metrics[f"rewards/{name}"].append(reward_per_func[i].item())
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_logps,
            "ref_per_token_logps": ref_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": full_mm_token_type_ids,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs: raise ValueError("GRPO does not support returning outputs")
        buffer_idx = self._step % self.args.gradient_accumulation_steps
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[buffer_idx] = inputs
        else:
            buffered_inputs = self._buffered_inputs[buffer_idx]
            if buffered_inputs is None:
                inputs = self._generate_and_score_completions(inputs, model)
                self._buffered_inputs[buffer_idx] = inputs
            else:
                inputs = buffered_inputs
        self._step += 1
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        mm_token_type_ids = inputs.get("mm_token_type_ids", None)
        if mm_token_type_ids is None and inputs["image_grid_thw"] is not None:
            mm_token_type_ids = self._build_mm_token_type_ids_from_input_ids(input_ids)
        logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            mm_token_type_ids,
        )[:, prompt_ids.size(1)-1:]
        old_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else logps.detach()
        adv = inputs["advantages"]
        coef1 = torch.exp(logps - old_logps)
        coef2 = torch.clamp(coef1, 1 - self.epsilon, 1 + self.epsilon)
        loss1 = coef1 * adv.unsqueeze(1)
        loss2 = coef2 * adv.unsqueeze(1)
        per_token_loss = -torch.min(loss1, loss2)
        if self.beta > 0:
            ref = inputs["ref_per_token_logps"]
            kl = torch.exp(ref - logps) - (ref - logps) - 1
            per_token_loss = per_token_loss + self.beta * kl
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(((kl * completion_mask).sum(1) / completion_mask.sum(1)).mean()).mean().item())
        loss = ((per_token_loss * completion_mask).sum(1) / completion_mask.sum(1)).mean()
        clipped = (loss1 < loss2).float()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics((clipped * completion_mask).sum() / completion_mask.sum()).mean().item())
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None):
        metrics = {k: sum(v) / len(v) for k, v in self._metrics.items() if len(v) > 0}
        logs = {**logs, **metrics}
        if version.parse(__import__('transformers').__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        effective = self.args.per_device_train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        return RepeatRandomSampler(ds, mini_repeat_count=self.num_generations, batch_size=effective // self.num_generations, repeat_count=self.num_iterations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return RepeatRandomSampler(eval_dataset, mini_repeat_count=self.num_generations, seed=self.args.seed)


def build_peft_config(args):
    if not args.use_lora:
        return None
    if LoraConfig is None:
        raise ImportError("peft is required for --use_lora")
    targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    return LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias=args.lora_bias, target_modules=targets, task_type="CAUSAL_LM")
