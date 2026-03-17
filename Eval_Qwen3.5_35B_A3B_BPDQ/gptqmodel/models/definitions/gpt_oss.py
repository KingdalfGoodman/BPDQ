# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import torch
import torch.nn.functional as F
from torch import nn

from ..base import BaseQModel


class GptOssExpertsNew(nn.Module):
    def __init__(self, config, ori_experts=None):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.alpha = 1.702
        self.limit = 7.0
        self.quantizing = False

        self.gate_up = nn.ModuleList([
            nn.Linear(self.hidden_size, 2 * self.expert_dim, dtype=config.dtype)
            for _ in range(self.num_experts)
        ])

        self.down = nn.ModuleList([
            nn.Linear(self.expert_dim, self.hidden_size, dtype=config.dtype)
            for _ in range(self.num_experts)
        ])

        if ori_experts is not None:
            self.quantizing = True
            for i in range(self.num_experts):
                tgt_gu_w = self.gate_up[i].weight   # [2E, H]
                tgt_gu_b = self.gate_up[i].bias     # [2E]
                tgt_d_w  = self.down[i].weight      # [H, E]
                tgt_d_b  = self.down[i].bias        # [H]

                gu_w_src = ori_experts.gate_up_proj[i].detach().t().contiguous()
                gu_b_src = ori_experts.gate_up_proj_bias[i].detach()
                d_w_src  = ori_experts.down_proj[i].detach().t().contiguous()
                d_b_src  = ori_experts.down_proj_bias[i].detach()

                with torch.inference_mode():
                    tgt_gu_w.copy_(gu_w_src)
                    tgt_gu_b.copy_(gu_b_src)
                    tgt_d_w.copy_(d_w_src)
                    tgt_d_b.copy_(d_b_src)

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        """
        库开发者的错误假设： gptqmodel 的开发者以为 routing_weights 永远是一个 2D 张量 (batch_size * seq_len, num_experts)。如果按他的设想，.shape[1] 确实应该等于 32。
        真实的残酷现实： 在初始化量化环境时，原始模型的 Router 传入的其实是一个 3D 张量 (batch_size, seq_len, num_experts)，即 (4, 979, 32)。
        """
        # ====== 1. 记录原始形状，防止出口时发生残差维度错位 ======
        orig_shape = hidden_states.shape

        # ====== 2. 核心架构修复：统一路由信息的维度 ======
        # 【修改点】：全部替换为 .reshape()，防止 torch.topk 带来的内存不连续问题
        if routing_weights is not None and routing_weights.dim() == 3:
            routing_weights = routing_weights.reshape(-1, routing_weights.shape[-1])
        if router_indices is not None and router_indices.dim() == 3:
            router_indices = router_indices.reshape(-1, router_indices.shape[-1])
        # ============================================
        # if self.quantizing:
        #     # For quantization, we need to trigger computation of all experts
        #     batch_size = hidden_states.shape[0]
        #     hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        #     num_experts = routing_weights.shape[1]

        #     hidden_states = hidden_states.repeat(num_experts, 1)
        #     hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
        #     gate_up = torch.stack([proj(hidden_states[i]) for i, proj in enumerate(self.gate_up)])
        #     gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        #     gate = gate.clamp(min=None, max=self.limit)
        #     up = up.clamp(min=-self.limit, max=self.limit)
        #     glu = gate * torch.sigmoid(gate * self.alpha)
        #     next_states = torch.stack([proj((up[i] + 1) * glu[i]) for i, proj in enumerate(self.down)])
        #     next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        #     next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
        #     next_states = next_states.sum(dim=0)

        #     return next_states

        if self.quantizing:
            # For quantization, we need to trigger computation of all experts
            # 【修改点】：不再依赖易出错的 [0] 索引，而是直接用总 token 数推导
            total_tokens = hidden_states.numel() // self.hidden_size
            hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (total_tokens, hidden_size)
            num_experts = routing_weights.shape[1]

            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            
            gate_up = torch.stack([proj(hidden_states[i]) for i, proj in enumerate(self.gate_up)])
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            
            next_states = torch.stack([proj((up[i] + 1) * glu[i]) for i, proj in enumerate(self.down)])
            
            # 【修改点】：基于严谨的 total_tokens 还原，杜绝批次/序列推算错误
            next_states = next_states.view(num_experts, total_tokens, self.hidden_size)
            
            # 这里 routing_weights 也被视为 (total_tokens, num_experts)
            weights_transposed = routing_weights.transpose(0, 1).unsqueeze(-1) # (num_experts, total_tokens, 1)
            next_states = next_states * weights_transposed
            
            next_states = next_states.sum(dim=0)
            
            # ====== 3. 强制恢复原始形状 ======
            return next_states.reshape(orig_shape)
        

        

        # For non-quantization forward pass, reduce forward pass time by only computing active experts
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1] if len(hidden_states.shape) > 2 else 1
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)

        active_experts = torch.unique(router_indices.flatten())
        final_output = torch.zeros_like(hidden_states)
        for expert_idx in active_experts:
            expert_mask = (router_indices == expert_idx).any(dim=-1)  # (num_tokens,)
            if not expert_mask.any():
                continue

            expert_tokens = hidden_states[expert_mask]  # (selected_tokens, hidden_size)

            gate_up_output = self.gate_up[expert_idx](expert_tokens)  # (selected_tokens, 2*expert_dim)
            gate, up = gate_up_output[..., ::2], gate_up_output[..., 1::2]

            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)

            expert_output = self.down[expert_idx]((up + 1) * glu)  # (selected_tokens, hidden_size)

            expert_weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)  # (selected_tokens, 1)

            final_output[expert_mask] += expert_output * expert_weights

        if seq_len > 1:
            final_output = final_output.view(batch_size, seq_len, self.hidden_size)
        else:
            final_output = final_output.view(batch_size, self.hidden_size)

        return final_output

class GptOssTopKRouterNew(nn.Module):
    def __init__(self, config, ori_router=None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

        if ori_router is not None:
            with torch.inference_mode():
                self.weight.copy_(ori_router.weight.detach())
                self.bias.copy_(ori_router.bias.detach())

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight.to(hidden_states.dtype), self.bias.to(hidden_states.dtype))  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        # return router_scores, router_indices
        return router_logits, router_scores, router_indices ###

class GPTOSSGPTQ(BaseQModel):
    dynamic_expert_index = "num_local_experts"

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "experts": {
                    "gate_up": {"#": ("#")},
                    "down": {"#": ("#")},
                }
            }
        }
    ]

    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_modeling

            gpt_oss_modeling.GptOssExperts = GptOssExpertsNew
            gpt_oss_modeling.GptOssTopKRouter = GptOssTopKRouterNew
