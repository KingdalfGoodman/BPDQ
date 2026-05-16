# BPDQ

> **BPDQ: Bit-Plane Decomposition Quantization on a Variable Grid for Large Language Models**

[![arXiv](https://img.shields.io/badge/arXiv-2602.04163-b31b1b.svg)](https://arxiv.org/abs/2602.04163)

Official implementation of BPDQ, a post-training quantization (PTQ) method that constructs a **variable quantization grid** via bit-plane decomposition and scalar coefficients, achieving high fidelity in low-bit regimes (2–3 bits) where conventional PTQ degrades sharply.

## Highlights

- **Variable grid via bit-planes.** BPDQ breaks the *shape invariance* of fixed grids, expanding the feasible set under the output-aligned objective.
- **Optimization-based.** All iterations stay aligned with the Hessian-induced geometry (output reconstruction objective), with formal consistency analysis in appendix.
- **Strong low-bit results.** 2-bit Qwen2.5-72B achieves **83.85% GSM8K** accuracy on a single RTX 3090 (22.69 GB), compared to 90.83% at 16-bit.


## Repository Structure

```
.
├── bpdq.patch                        # Patch on top of GPTQModel v5.7.0
├── bpdq_work_flow_git.py     # Quantize + evaluate workflow
├── config.yaml                       # Example BPDQ config
├── requirements.txt                # Reference dependencies
├── Eval_Qwen3.5_35B_A3B_BPDQ/  # Evaluation logs (Beta)
├── LICENSE
└── README.md
```


## Quantized Models

Selected BPDQ checkpoints are available on the Hugging Face Hub:

| Base Model | Bits / Group | Checkpoint |
|---|---|---|
| Qwen2.5-72B | 2-bit / g=256 | [Qwen2.5-72B-BPD2-G256](https://huggingface.co/goodman20241017/Qwen2.5-72B-BPD2-G256) |
| Llama-2-7B  | 2-bit / g=64  | [Llama-2-7B-BPD2-G64](https://huggingface.co/goodman20241017/Llama-2-7B-BPD2-G64) |
| Llama-2-7B  | 3-bit / g=64  | [Llama-2-7B-BPD3-G64](https://huggingface.co/goodman20241017/Llama-2-7B-BPD3-G64) |

> **Note on size.** The figure reported on Hugging Face (e.g., 24.2 GB for `Qwen2.5-72B-BPD2-G256`) is the **checkpoint file size on disk**, measured in GB (10⁹ bytes). The **22.69 GB** value cited in our paper refers to the **peak VRAM** during inference on a single RTX 3090 — a different measurement. Note also that Linux tools such as `du -h` display sizes in GiB (2³⁰ bytes), so `24.2 GB ≈ 22.54 GiB` on disk; this is a unit convention, not a discrepancy.





## Installation

BPDQ is implemented as a patch on top of [GPTQModel](https://github.com/ModelCloud/GPTQModel) at version **5.7.0**.

```bash
# 1. Clone GPTQModel and check out v5.7.0
git clone https://github.com/ModelCloud/GPTQModel.git
cd GPTQModel
git checkout v5.7.0

# 2. Apply the BPDQ patch (path it to wherever you cloned this repo)
git apply /path/to/BPDQ/bpdq.patch

# 3. Install (editable mode recommended)
pip install -e .
```

Other dependencies (`transformers`, `lm-evaluation-harness`, etc.) are listed in `requirements.txt` **for reference only** — the actual environment is somewhat opinionated and may require manual adjustment depending on your CUDA / PyTorch / system setup.

### Calibration Data

The workflow uses [C4](https://huggingface.co/datasets/allenai/c4) for calibration. Download a single training shard, e.g. `en.noblocklist/c4-train.00001-of-01024.json.gz`, and point `build_calibration_dataset()` in `bpdq_work_flow_git.py` to it:

```python
local_c4_file = "YOUR_PATH/c4-train.00001-of-01024.json.gz"
```

## Quick Start

Edit `config.yaml` with your local paths, then:

```bash
python bpdq_work_flow_git.py --config config.yaml
```

The script will:

1. **Quantize** each model in `models:` for every BPDQ configuration in `sweep.bpdq` (combinations of `msbits`, `group_sizes`, `n_iters`, `alpha`).
2. **Save** each quantized checkpoint under `paths.quant_root`.
3. **Evaluate** each checkpoint on every entry in `task_configs` (wikitext, gsm8k, mmlu, ...) via `lm-evaluation-harness`, with per-run statistics dumped to `paths.base_output_dir/run_stats_*.json`.

### Configuration

Minimal `config.yaml`:

```yaml
paths:
  model_root: YOUR_PATH/model
  quant_root: YOUR_PATH/quant_model
  base_output_dir: YOUR_PATH/eval_results/Qwen3-0.6B_xxx

models:
  - alias: Qwen3-0.6B_xxx
    pretrained: YOUR_PATH/model/Qwen3-0.6B

sweep:
  bpdq:
    w_bits: [8]               # For bitplane init., select the k most significant bit (MSB) planes.
    msbits: [4, 3, 2]       # number of bit-planes (controls effective BPW)
    group_sizes: [128]
    n_iters: [10]
    alpha: [1.0e-4]         # ⚠️ must be 1.0e-4 — YAML 1.1 parses `1e-4` as a string

task_configs:
  - {tasks: [wikitext], eval_batch_size: 2,  num_fewshot: 0}
  - {tasks: [gsm8k],    eval_batch_size: 32, num_fewshot: 5}
  # ... see config.yaml for the full task list
```

> Tip: run without `--config` to fall back to the in-script defaults.
>
> The script also contains `gptq` and `awq` modes in `SWEEP_CONFIG` for advanced users; only the **BPDQ** workflow is documented here.



### Evaluation Only

If you already have BPDQ-quantized checkpoints and just want to evaluate them without re-quantizing, switch on `eval_only` in your config and list the checkpoint paths:

```yaml
eval_only: true
eval_models:
  - YOUR_PATH/quant_model/xxx

paths:
  base_output_dir: YOUR_PATH/eval_results_xxx

task_configs:
  - {tasks: [wikitext], eval_batch_size: 2,  num_fewshot: 0}
  - {tasks: [gsm8k],    eval_batch_size: 32, num_fewshot: 5}

eval_defaults:
  device: cuda
  eval_dtype: bfloat16
  eval_trust_remote_code: true
```

Then run as usual:

```bash
python bpdq_work_flow_git.py --config config_eval_only.yaml
```

This skips quantization, going straight to evaluating each listed checkpoint on every task in `task_configs`. 

> When `eval_only: false` (or omitted), the script behaves identically to the full quantize-then-eval flow described above.



## Citation

If you find BPDQ useful in your research, please cite:

```bibtex
@article{chen2026bpdq,
  title={BPDQ: Bit-Plane Decomposition Quantization on a Variable Grid for Large Language Models},
  author={Chen, Junyu and Li, Jungang and Xiong, Jing and Wang, Wenjie and Yang, Qingyao and Xiao, He and Li, Zhen and Wu, Taiqiang and Chen, Mengzhao and Peng, Zhen and others},
  journal={arXiv preprint arXiv:2602.04163},
  year={2026}
}
```

## Contact

For any questions regarding BPDQ, please open an issue or contact:  
**kingdalfgoodman[at]foxmail[dot]com**

*Trivia: The lowercase `bpdq` is perfectly symmetrical.*