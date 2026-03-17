import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = ""
# os.environ["HF_DATASETS_CACHE"] = ""
# os.environ["HF_HUB_CACHE"] = "" 

import gc
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, replace
from typing import Dict, Any, List

import torch
from transformers import HfArgumentParser
# ====== 【 Tokenicer breakdown, FIX for Qwen3.5-MoE】======
import transformers
orig_getattr = transformers.PretrainedConfig.__getattribute__
def safe_getattr(self, key):
    try:
        return orig_getattr(self, key)
    except AttributeError:
        if key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            return None
        raise
transformers.PretrainedConfig.__getattribute__ = safe_getattr
#
# ====== 【FIX】Qwen3.5-MoE mix-attn Mask dims conflict ======
import torch
try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeAttention
    orig_attn_forward = Qwen3_5MoeAttention.forward

    def patched_attn_forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        if attention_mask is not None and attention_mask.dim() == 2:
            if attention_mask.min() == 1:
                attention_mask = None 
            else:
                seq_len = attention_mask.shape[-1]
                causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=attention_mask.device, dtype=torch.bool))
                pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
                attention_mask = (causal_mask & pad_mask)
                
        return orig_attn_forward(self, hidden_states, attention_mask=attention_mask, *args, **kwargs)

    Qwen3_5MoeAttention.forward = patched_attn_forward
    print("\n[PATCH SUCCESS]  Attention Fix Qwen3.5-MoE  Mask broadcast, and protect Padding!\n")
except ImportError:
    print("\n[WARN] Not find Qwen3_5MoeAttention, Fail. \n")
# ========================================

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

import logging
logger = logging.getLogger(__name__)


# =========================
TARGET_MODEL_PATH = ".../Qwen3.5-35B-A3B_BPD2_g128_als5_8"
TASK_CONFIGS_TO_RUN = [
    # {"tasks": ["wikitext"],         "eval_batch_size": 1,  "num_fewshot": 0},   
    # {"tasks": ["arc_challenge"],    "eval_batch_size": 32, "num_fewshot": 0,},  
    # {"tasks": ["boolq"],            "eval_batch_size": 64, "num_fewshot": 0,},  
    # {"tasks": ["minerva_math500"],  "eval_batch_size": 64, "num_fewshot": 4,},  
    # {"tasks": ["gsm8k"],            "eval_batch_size": 64, "num_fewshot": 5,}, 
    # {"tasks": ["hellaswag"],        "eval_batch_size": 64, "num_fewshot": 0,}, 
    # {"tasks": ["mmlu"],             "eval_batch_size": 8,  "num_fewshot": 0,}, 
]
BASE_OUTPUT_DIR = ".../eval_results"
RUN_STATS: List[Dict[str, Any]] = []
STATS_JSON_PATH = os.path.join(BASE_OUTPUT_DIR, f"run_stats_{datetime.now().strftime("%d_%H%M")}.json")


# =========================
@dataclass
class EvalConfig:
    tasks: List[str] = field(default_factory=lambda: ["wikitext"])
    eval_batch_size: int = 2
    num_fewshot: int = 0
    model_path: str = TARGET_MODEL_PATH
    output_path: str = "./eval_results/"
    trust_remote_code: bool = True
    device: str = "cuda"
    dtype: str = "auto"  


# =========================
def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        _cuda_sync()

def _get_cuda_peak_mib() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    _cuda_sync()
    total_alloc = 0.0
    total_rsv = 0.0
    for i in range(torch.cuda.device_count()):
        total_alloc += torch.cuda.max_memory_allocated(i) / (1024 ** 2)
        total_rsv += torch.cuda.max_memory_reserved(i) / (1024 ** 2)
    return {"peak_allocated_mib": float(total_alloc), "peak_reserved_mib": float(total_rsv)}

def _dump_stats():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    with open(STATS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(RUN_STATS, f, ensure_ascii=False, indent=2)

def print_cuda_mem(prefix: str = ""):
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        free_memory, total_memory = torch.cuda.mem_get_info(device_id)
        free_gb = free_memory / (1024 ** 3)
        total_gb = total_memory / (1024 ** 3)
        print(f"{prefix}PyTorch device: cuda:{device_id}")
        print(f"{prefix}Total: {total_gb:.2f} GB")
        print(f"{prefix}available: {free_gb:.2f} GB")

def save_results(results, eval_args: EvalConfig):
    os.makedirs(eval_args.output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%d_%H%M")
    results_wo_samples = {k: v for k, v in results.items() if k != "samples"}

    model_name = os.path.basename(eval_args.model_path.rstrip("/"))
    tasks_str = "_".join(eval_args.tasks)

    full_results = {
        "timestamp": timestamp,
        "model_name": model_name,
        "arguments": {"eval_args": vars(eval_args)},
        "evaluation_results": results_wo_samples,
    }

    results_file = os.path.join(
        eval_args.output_path,
        f"{model_name}_{tasks_str}_{timestamp}.json",
    )
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")


def run_evaluation(eval_args: EvalConfig):
    print(f"\n--- EVAL ---")
    print(f"Model Path: {eval_args.model_path}")
    print(f"Task: {eval_args.tasks}")
    print(f"Batch Size: {eval_args.eval_batch_size}")
    print(f"Shot: {eval_args.num_fewshot}")

    _reset_cuda_peak()
    t0 = time.perf_counter()
    _cuda_sync()

    hflm_kwargs = dict(
        pretrained=eval_args.model_path,
        trust_remote_code=eval_args.trust_remote_code,
        dtype=eval_args.dtype,
        batch_size=eval_args.eval_batch_size,
        device_map="auto",
        gptqmodel=True, 
    )

    lm = HFLM(**hflm_kwargs)
 
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=eval_args.tasks,
        num_fewshot=eval_args.num_fewshot,
        batch_size=eval_args.eval_batch_size,
    )

    _cuda_sync()
    elapsed = time.perf_counter() - t0
    mem = _get_cuda_peak_mib()

    print("--- Evaluation completed ---")
    if "groups" in results:
        print(json.dumps(results["groups"], indent=2))
    print(make_table(results))

    if eval_args.output_path:
        save_results(results, eval_args)

    task_str = "_".join(eval_args.tasks)
    model_basename = os.path.basename(eval_args.model_path.rstrip("/"))

    print(
        f"[EVAL-STAT] model={model_basename} | tasks={task_str} | fewshot={eval_args.num_fewshot} | "
        f"bs={eval_args.eval_batch_size} | time={elapsed:.2f}s | "
        f"peak_alloc={mem['peak_allocated_mib']:.1f} MiB | peak_reserved={mem['peak_reserved_mib']:.1f} MiB"
    )

    RUN_STATS.append({
        "stage": "eval",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": eval_args.model_path,
        "model_name": model_basename,
        "tasks": eval_args.tasks,
        "num_fewshot": eval_args.num_fewshot,
        "eval_batch_size": eval_args.eval_batch_size,
        "elapsed_sec": float(elapsed),
        **mem,
    })

    _dump_stats()
    return results


if __name__ == "__main__":
    print_cuda_mem(prefix="[INIT] ")

    parser_e = HfArgumentParser(EvalConfig)
    base_eval_args = parser_e.parse_args_into_dataclasses()[0]

    model_tag = os.path.basename(TARGET_MODEL_PATH.rstrip("/"))
    current_timestamp = datetime.now().strftime("%d_%H%M")
    current_output_path = os.path.join(BASE_OUTPUT_DIR, f"{model_tag}_{current_timestamp}")
    print(f"\n========== STARTING EVALUATION FOR: {model_tag} ==========")

    for task_config in TASK_CONFIGS_TO_RUN:
        current_eval = replace(
            base_eval_args,
            output_path=current_output_path,
            **task_config,
        )
        task_name_str = "_".join(current_eval.tasks)
        print(f"\n--- Running evaluation: [MODEL: {model_tag}] [TASK: {task_name_str}] ---")

        try:
            run_evaluation(current_eval)
            print(f"--- Evaluation completed: [MODEL: {model_tag}] [TASK: {task_name_str}] ---")
        except Exception as e:
            import traceback
            print(f"--- [FAIL] [MODEL: {model_tag}] [TASK: {task_name_str}] ---")
            print(f"ERROR: {e}")
            print("\n--- Traceback ---")
            print(traceback.format_exc())
            print("---------------------------------")

        gc.collect()
        torch.cuda.empty_cache()

    print("\n--- All evaluation have been completed ---")