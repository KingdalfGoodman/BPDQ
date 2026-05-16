import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

os.environ["HF_HOME"] = "YOUR_PATH/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "YOUR_PATH/.cache/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = "YOUR_PATH/.cache/huggingface/hub" 

import gc
import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Dict, Any, List
import time

import torch
from datasets import load_dataset
from transformers import HfArgumentParser

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import METHOD, FORMAT, BPDQConfig

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

logger = logging.getLogger(__name__)


import argparse
import yaml

# ----- 读取 yaml(不传 --config 也能跑,用 yaml 内的默认或下面写死的回落) -----
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("--config", default=None)
_cli_args, _ = _cli.parse_known_args()   # 用 parse_known_args 不影响后面 HfArgumentParser

if _cli_args.config:
    with open(_cli_args.config, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) or {}
    print(f"[CONFIG] Loaded {_cli_args.config}")
else:
    _cfg = {}
    print("[CONFIG] No --config given, using built-in defaults")

# ----- 路径 -----
_paths = _cfg.get("paths", {})
MODEL_ROOT       = _paths.get("model_root",       "YOUR_PATH/model")
QUANT_ROOT       = _paths.get("quant_root",       "YOUR_PATH/model/quant_model")
BASE_OUTPUT_DIR  = _paths.get("base_output_dir",  "YOUR_PATH/coding/eval_results/0514_Qwen3-0.6B")

# ----- 模型列表 / 扫描 / 任务 -----
MODELS_TO_RUN = _cfg.get("models", [
    {"alias": "0514_Qwen3-0.6B", "pretrained": f"{MODEL_ROOT}/Qwen3-0.6B"},
])

SWEEP_CONFIG = _cfg.get("sweep", {
    "bpdq": {
        "w_bits": [8], "msbits": [4, 3, 2], "group_sizes": [64],
        "n_iters": [10], "alpha": [1e-4],
    },
})
MODE_CHOICES = [m for m, c in SWEEP_CONFIG.items() if c]

TASK_CONFIGS_TO_RUN = _cfg.get("task_configs", [
    {"tasks": ["wikitext"],        "eval_batch_size": 2,  "num_fewshot": 0},
    {"tasks": ["arc_challenge"],   "eval_batch_size": 32, "num_fewshot": 0},
    {"tasks": ["boolq"],           "eval_batch_size": 32, "num_fewshot": 0},
    {"tasks": ["minerva_math500"], "eval_batch_size": 32, "num_fewshot": 4},
    {"tasks": ["gsm8k"],           "eval_batch_size": 32, "num_fewshot": 5},
    {"tasks": ["hellaswag"],       "eval_batch_size": 32, "num_fewshot": 0},
    {"tasks": ["mmlu"],            "eval_batch_size": 4,  "num_fewshot": 0},
])

RUN_STATS: List[Dict[str, Any]] = []
STATS_JSON_PATH = os.path.join(
    BASE_OUTPUT_DIR, f"run_stats_{datetime.now().strftime('%d_%H%M')}.json"
)


_MODEL_DEFAULTS = _cfg.get("model_defaults", {})
_EVAL_DEFAULTS  = _cfg.get("eval_defaults", {})
EVAL_ONLY   = bool(_cfg.get("eval_only", False))
EVAL_MODELS = _cfg.get("eval_models", [])


@dataclass
class ModelConfig:
    pretrained: str = ""
    sym: bool = False

    w_bits: int = 8
    group_size: int = 128
    quantized_model_root: str = QUANT_ROOT

    device_map: str = "auto"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True


class ModelConfigManager:
    def __init__(self, args: ModelConfig):
        self.args = args

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {
            "torch_dtype": torch.bfloat16,
            "device_map": self.args.device_map,
            "trust_remote_code": self.args.trust_remote_code,
        }


@dataclass
class EvalConfig:
    tasks: List[str] = field(default_factory=lambda: ["commonsense_qa"])
    eval_batch_size: int = 64
    num_fewshot: int = 0

    model_path: str = ""
    output_path: str = "./eval_results/"

    device: str = "cuda"
    eval_trust_remote_code: bool = True
    eval_dtype: str = "bfloat16"   # "auto", "float16", "bfloat16"


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
        print(f"{prefix}Current PyTorch visible device: cuda:{device_id}")
        print(f"{prefix}Total VRAM: {total_gb:.2f} GB")
        print(f"{prefix}Available VRAM: {free_gb:.2f} GB")


def build_calibration_dataset() -> List[str]:
    local_c4_file = "YOUR_PATH/model/datasets/c4_local/en.noblocklist/c4-train.00001-of-01024.json.gz"
    ds = load_dataset("json", data_files={"train": local_c4_file}, split="train").select(range(1024))
    return ds["text"]


def run_one_quant(
    model_args: ModelConfig,
    quant_config: QuantizeConfig,
    save_dir: str,
    save_name: str,
    calibration_dataset: List[str],
) -> str:
    manager = ModelConfigManager(model_args)

    _reset_cuda_peak()
    model = GPTQModel.load(
        model_args.pretrained,
        **manager.get_model_kwargs(),
        quantize_config=quant_config,
    )
    _cuda_sync()
    t0 = time.perf_counter()
    model.quantize(calibration_dataset, batch_size=1)

    _cuda_sync()
    elapsed = time.perf_counter() - t0
    mem = _get_cuda_peak_mib()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    print(f"[SAVE] {save_path}")
    model.save(save_path)

    print(
        f"[QUANT-STAT] name={save_name} | time={elapsed:.2f}s | "
        f"peak_alloc={mem['peak_allocated_mib']:.1f} MiB | peak_reserved={mem['peak_reserved_mib']:.1f} MiB"
    )
    
    is_bpdq = (quant_config.quant_method == METHOD.BPDQ and quant_config.bpdq is not None)
    RUN_STATS.append({
        "stage": "quant",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "save_name": save_name,
        "model_pretrained": model_args.pretrained,
        "quantized_path": save_path,
        "quant_mode_bits": getattr(model_args, "w_bits", None),
        "group_size": getattr(model_args, "group_size", None),
        "elapsed_sec": float(elapsed),
        "msbits": quant_config.bpdq.msbits if is_bpdq else None,
        "n_iters": quant_config.bpdq.n_iters if is_bpdq else None,
        "alpha": quant_config.bpdq.alpha if is_bpdq else None,
        **mem,
    })

    _dump_stats()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return save_path


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
    print(f"--- Starting model evaluation ---")
    print(f"Model path: {eval_args.model_path}")
    print(f"Tasks: {eval_args.tasks}")
    print(f"Batch Size: {eval_args.eval_batch_size}")
    print(f"Shot: {eval_args.num_fewshot}")

    _reset_cuda_peak()
    t0 = time.perf_counter()
    _cuda_sync()

    hflm_kwargs = dict(
        pretrained=eval_args.model_path,
        trust_remote_code=eval_args.eval_trust_remote_code,
        dtype=eval_args.eval_dtype,
        device=eval_args.device,
        batch_size=eval_args.eval_batch_size,
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

    print("--- Evaluation complete ---")
    if "groups" in results:
        print("--- Aggregated results (Groups) ---")
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

    base_model_args = ModelConfig(**_MODEL_DEFAULTS)
    base_eval_args  = EvalConfig(**_EVAL_DEFAULTS)

    if EVAL_ONLY:
        if not EVAL_MODELS:
            raise ValueError("eval_only=true but eval_models is empty in config")
        print(f"\n========== EVAL-ONLY MODE ({len(EVAL_MODELS)} models) ==========")
        for save_path in EVAL_MODELS:
            model_tag = os.path.basename(save_path.rstrip("/"))
            current_timestamp = datetime.now().strftime("%d_%H%M")
            current_output_path = os.path.join(BASE_OUTPUT_DIR, f"{model_tag}_{current_timestamp}")
            for task_config in TASK_CONFIGS_TO_RUN:
                current_eval = replace(
                    base_eval_args,
                    model_path=save_path,
                    output_path=current_output_path,
                    **task_config,
                )
                task_name_str = "_".join(current_eval.tasks)
                print(f"\n--- Running evaluation: [Model: {model_tag}] [Task: {task_name_str}] ---")
                try:
                    run_evaluation(current_eval)
                    print(f"--- Successfully completed evaluation: [Model: {model_tag}] [Task: {task_name_str}] ---")
                except Exception as e:
                    import traceback
                    print(f"--- [FAILED] [Model: {model_tag}] [Task: {task_name_str}] ---")
                    print(f"Error message: {e}")
                    print("\n--- Full error info (Traceback) ---")
                    print(traceback.format_exc())
                    print("---------------------------------")
                gc.collect()
                torch.cuda.empty_cache()
        print("\n--- Evaluation-only runs finished ---")
        import sys
        sys.exit(0)

    calibration_dataset = build_calibration_dataset()
    for quant_mode in MODE_CHOICES:
        print(f"\n========== QUANT MODE: {quant_mode.upper()} ==========")
        cfg = SWEEP_CONFIG[quant_mode]

        for model_info in MODELS_TO_RUN:
            alias = model_info["alias"]
            pretrained = model_info["pretrained"]
            base_model_name = os.path.basename(pretrained.rstrip("/"))

            save_dir = os.path.join(QUANT_ROOT, f"{alias}_{quant_mode}")
            print(f"\n--- Model: {base_model_name} ({quant_mode}) ---")

            model_base_args = replace(base_model_args, pretrained=pretrained)

            produced_model_paths: List[str] = []

            if quant_mode == "gptq":
                w_bits_list = cfg["w_bits"]
                group_sizes = cfg["group_sizes"]

                for w_bits in w_bits_list:
                    for group_size in group_sizes:
                        args = replace(model_base_args, w_bits=w_bits, group_size=group_size)
                        quant_cfg = QuantizeConfig(
                            bits=w_bits,
                            group_size=group_size,
                            sym=args.sym,
                            desc_act=True,
                        )
                        save_name = f"{base_model_name}_INT{w_bits}_g{group_size}"
                        save_path = run_one_quant(args, quant_cfg, save_dir, save_name, calibration_dataset)
                        produced_model_paths.append(save_path)

            elif quant_mode == "bpdq":
                w_bits_list = cfg["w_bits"]
                msbits = cfg["msbits"]
                group_sizes = cfg["group_sizes"]
                n_iters_list = cfg["n_iters"]
                alpha_list = cfg.get("alpha", [1e-4])


                for w_bits in w_bits_list:
                    for msbit in msbits:
                        for group_size in group_sizes:
                            for n_iters in n_iters_list:
                                for alpha in alpha_list: 
                                    args = replace(
                                        model_base_args,
                                        w_bits=w_bits,
                                        group_size=group_size,
                                    )
                                    bpdq_cfg = BPDQConfig(
                                        msbits=msbit,
                                        n_iters=n_iters,
                                        alpha=alpha
                                    )
                                    quant_cfg = QuantizeConfig(
                                        bits=w_bits,
                                        group_size=group_size,
                                        sym=args.sym,
                                        quant_method=METHOD.BPDQ, 
                                        format=FORMAT.BPDQ,       
                                        bpdq=bpdq_cfg,            
                                        desc_act=False,
                                        act_group_aware=True,
                                        offload_to_disk=False,  
                                    )

                                    alpha_tag = f"a{alpha:.0e}".replace("+", "")
                                    save_name = f"{base_model_name}_BPD{msbit}_g{group_size}_als{n_iters}_{w_bits}_{alpha_tag}"
                                    save_path = run_one_quant(args, quant_cfg, save_dir, save_name, calibration_dataset)
                                    produced_model_paths.append(save_path)

            elif quant_mode == "awq":  
                w_bits_list = cfg["w_bits"]  
                group_sizes = cfg["group_sizes"]  
                awq_formats = cfg.get("formats", ["gemm"])  
                for fmt_name in awq_formats:  
                    fmt_key = str(fmt_name).lower().strip()
                    awq_format_enum = FORMAT.GEMM  
                    for w_bits in w_bits_list:  
                        for group_size in group_sizes:  
                            args = replace(model_base_args, w_bits=w_bits, group_size=group_size)  
                            quant_cfg = QuantizeConfig(  
                                bits=w_bits,  
                                group_size=group_size,  
                                sym=args.sym,  
                                quant_method=METHOD.AWQ,  
                                format=awq_format_enum,  
                            )  
                            save_name = f"{base_model_name}_AWQ{fmt_key.upper()}_INT{w_bits}_g{group_size}"  
                            save_path = run_one_quant(args, quant_cfg, save_dir, save_name, calibration_dataset)  
                            produced_model_paths.append(save_path)  
            else:
                raise ValueError(f"Unknown quant_mode: {quant_mode}")

            for save_path in produced_model_paths:
                model_tag = os.path.basename(save_path.rstrip("/"))
                current_timestamp = datetime.now().strftime("%d_%H%M")
                current_output_path = os.path.join(BASE_OUTPUT_DIR, f"{model_tag}_{current_timestamp}")

                for task_config in TASK_CONFIGS_TO_RUN:
                    current_eval = replace(
                        base_eval_args,
                        model_path=save_path,          
                        output_path=current_output_path,
                        **task_config,
                    )
                    task_name_str = "_".join(current_eval.tasks)
                    model_tag = os.path.basename(save_path.rstrip("/"))
                    print(f"\n--- Running evaluation: [Model: {model_tag}] [Task: {task_name_str}] ---")

                    try:
                        run_evaluation(current_eval)
                        print(f"--- Successfully completed evaluation: [Model: {model_tag}] [Task: {task_name_str}] ---")
                    except Exception as e:
                        import traceback
                        print(f"--- [FAILED] [Model: {model_tag}] [Task: {task_name_str}] ---")
                        print(f"Error message: {e}")
                        print("\n--- Full error info (Traceback) ---")
                        print(traceback.format_exc())
                        print("---------------------------------")

                    gc.collect()
                    torch.cuda.empty_cache()

    print("\n--- All quantization and evaluation runs finished ---")
