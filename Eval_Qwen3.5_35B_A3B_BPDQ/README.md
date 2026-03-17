The currently released `Eval_Qwen3.5_35B_A3B_BPDQ` code is intended solely for testing `Qwen3.5-35B-A3B-BPD2-g128`. 

**How to run:**
1. Install GPT-QModel v5.7.0.
2. Overwrite the existing `gptqmodel` folder with the `gptqmodel` folder located in the `Eval_Qwen3.5_35B_A3B_BPDQ` directory.
3. Run `qwen35_eval.py` to perform the model evaluation.

For the versions of other related libraries, please refer to `requirements.txt`.

* **Paper:** https://arxiv.org/abs/2602.04163
* **HF:** https://huggingface.co/goodman20241017/Qwen3.5-35B-A3B-BPD2-g128
**Note:** Not yet specifically optimized for MoE models. This is a preliminary experimental version.