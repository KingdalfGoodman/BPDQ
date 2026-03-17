# gptqmodel/nn_modules/qlinear/bpdq_torch.py

import math
import torch
import torch.nn as nn
from typing import List

from . import BaseQuantLinear
from ...models._const import DEVICE, PLATFORM
from ...quantization.config import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...adapter.adapter import Adapter


import triton
import triton.language as tl
@triton.jit
def bpdq_kernel(
    # Pointers
    x_ptr, B_ptr, alpha_ptr, bias_ptr, out_ptr, Linear_bias_ptr, # Linear_bias_ptr from model
    # Shapes
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_bk, stride_bb, stride_bn,
    stride_ag, stride_ab, stride_an,
    stride_bg, stride_bn_bias,
    stride_om, stride_on, stride_ob,  # Linear_bias_ptr from model
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,      # == GROUP_SIZE
    GROUP_SIZE: tl.constexpr,   # == BLOCK_K
    N_PLANES: tl.constexpr,     # bit-planes: 1,2,3,4
    HAS_LINEAR_BIAS: tl.constexpr,  # Linear_bias_ptr from model
):
    tl.static_assert(GROUP_SIZE % 32 == 0, "GROUP_SIZE must be divisible by 32")
    tl.static_assert(BLOCK_K == GROUP_SIZE, "BLOCK_K must equal GROUP_SIZE in this kernel")
    tl.static_assert(BLOCK_K % 32 == 0, "BLOCK_K must be divisible by 32")
    tl.static_assert(N_PLANES >= 1, "N_PLANES must be >= 1")
    tl.static_assert(N_PLANES <= 4, "N_PLANES must be <= 4")

    stride_xm = tl.full((), stride_xm, tl.int64)
    stride_xk = tl.full((), stride_xk, tl.int64)

    stride_bk = tl.full((), stride_bk, tl.int64)
    stride_bb = tl.full((), stride_bb, tl.int64)
    stride_bn = tl.full((), stride_bn, tl.int64)

    stride_ag = tl.full((), stride_ag, tl.int64)
    stride_ab = tl.full((), stride_ab, tl.int64)
    stride_an = tl.full((), stride_an, tl.int64)

    stride_bg = tl.full((), stride_bg, tl.int64)
    stride_bn_bias = tl.full((), stride_bn_bias, tl.int64)

    stride_om = tl.full((), stride_om, tl.int64)
    stride_on = tl.full((), stride_on, tl.int64)
    stride_ob = tl.full((), stride_ob, tl.int64)

    pid = tl.program_id(axis=0).to(tl.int64)

    # ---------------- Grid Swizzle (L2 Cache optimal) ----------------
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    GROUP_M = 8
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    # pid_m = first_pid_m + (pid % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    # -----------------------------------------------------------

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64) # int32 will ERROR in qwen 30b 
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)


    valid_m = offs_m < M                      # [BLOCK_M]
    valid_n = offs_n < N                      # [BLOCK_N]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_ptr_base = x_ptr + offs_m[:, None] * stride_xm

    bn_bias = offs_n * stride_bn_bias        # [BLOCK_N]
    an      = offs_n * stride_an             # [BLOCK_N]
    bn      = offs_n * stride_bn             # [BLOCK_N]

    shift_vals = tl.arange(0, 32)[:, None]
    k_offsets = tl.arange(0, 32)[None, :]

    for k_block_start in range(0, K, BLOCK_K):
        current_group_idx = (k_block_start // GROUP_SIZE)
        
        bias_offset = current_group_idx * stride_bg + bn_bias
        bias_val = tl.load(bias_ptr + bias_offset, mask=valid_n, other=0.0).to(tl.float32)  

        if N_PLANES >= 1:
            a_off_0 = current_group_idx * stride_ag + 0 * stride_ab + an
            alpha_0 = tl.load(alpha_ptr + a_off_0, mask=valid_n, other=0.0).to(tl.float32) 
        if N_PLANES >= 2:
            a_off_1 = current_group_idx * stride_ag + 1 * stride_ab + an
            alpha_1 = tl.load(alpha_ptr + a_off_1, mask=valid_n, other=0.0).to(tl.float32)
        if N_PLANES >= 3:
            a_off_2 = current_group_idx * stride_ag + 2 * stride_ab + an
            alpha_2 = tl.load(alpha_ptr + a_off_2, mask=valid_n, other=0.0).to(tl.float32)
        if N_PLANES >= 4:
            a_off_3 = current_group_idx * stride_ag + 3 * stride_ab + an
            alpha_3 = tl.load(alpha_ptr + a_off_3, mask=valid_n, other=0.0).to(tl.float32)


        for k_sub_idx in tl.static_range(0, BLOCK_K, 32):
            current_k = k_block_start + k_sub_idx


            k_range = (current_k + k_offsets).to(tl.int64)  # [1, 32]
            x_ptrs = x_ptr_base + k_range * stride_xk
            x_chunk = tl.load(x_ptrs, mask=valid_m[:, None], other=0.0).to(tl.bfloat16)


            k_tile = (current_k // 32)              

            w_delta = tl.zeros((32, BLOCK_N), dtype=tl.float32)

            # plane 0
            if N_PLANES >= 1:
                b_off_0 = k_tile * stride_bk + 0 * stride_bb + bn
                packed_0 = tl.load(B_ptr + b_off_0, mask=valid_n, other=0)   # int32
                bits_0 = ((packed_0[None, :] >> shift_vals) & 1)
                w_delta += bits_0.to(tl.float32) * alpha_0[None, :]

            # plane 1
            if N_PLANES >= 2:
                b_off_1 = k_tile * stride_bk + 1 * stride_bb + bn
                packed_1 = tl.load(B_ptr + b_off_1, mask=valid_n, other=0)
                bits_1 = ((packed_1[None, :] >> shift_vals) & 1)
                w_delta += bits_1.to(tl.float32) * alpha_1[None, :]

            # plane 2
            if N_PLANES >= 3:
                b_off_2 = k_tile * stride_bk + 2 * stride_bb + bn
                packed_2 = tl.load(B_ptr + b_off_2, mask=valid_n, other=0)
                bits_2 = ((packed_2[None, :] >> shift_vals) & 1)
                w_delta += bits_2.to(tl.float32) * alpha_2[None, :]

            # plane 3
            if N_PLANES >= 4:
                b_off_3 = k_tile * stride_bk + 3 * stride_bb + bn
                packed_3 = tl.load(B_ptr + b_off_3, mask=valid_n, other=0)
                bits_3 = ((packed_3[None, :] >> shift_vals) & 1)
                w_delta += bits_3.to(tl.float32) * alpha_3[None, :]

            w_chunk = (w_delta + bias_val[None, :]).to(tl.bfloat16)          # [32, BLOCK_N]

            # x_chunk: [BLOCK_M, 32], w_chunk: [32, BLOCK_N]
            accumulator += tl.dot(x_chunk, w_chunk)

    if HAS_LINEAR_BIAS:  # __add_out_bias_apply__
        ob_offs = offs_n * stride_ob  # __add_out_bias_apply__
        out_b = tl.load(Linear_bias_ptr + ob_offs, mask=valid_n, other=0.0).to(tl.float32)  # __add_out_bias_apply__
        accumulator += out_b[None, :]  # __add_out_bias_apply__

    c_mask = valid_m[:, None] & valid_n[None, :]
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)

# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bpdq_kernel_v2(
    # Pointers
    x_ptr, B_ptr, alpha_ptr, bias_ptr, out_ptr, Linear_bias_ptr,
    # Shapes
    M, N, K,
    # Strides (Int64)
    stride_xm, stride_xk,
    stride_bk, stride_bb, stride_bn,
    stride_ag, stride_ab, stride_an,
    stride_bg, stride_bn_bias,
    stride_om, stride_on, stride_ob,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,      
    GROUP_SIZE: tl.constexpr,    
    N_PLANES: tl.constexpr,      
    HAS_LINEAR_BIAS: tl.constexpr,
):
    tl.static_assert(GROUP_SIZE % 32 == 0, "GROUP_SIZE must be divisible by 32")
    tl.static_assert(BLOCK_K == GROUP_SIZE, "BLOCK_K must equal GROUP_SIZE for this logic")

    # 2. (Int32 Logic -> Int64 Pointer)
    pid = tl.program_id(axis=0)
    
    # Grid Swizzle
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    GROUP_M = 8
    num_pid_in_group = GROUP_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    valid_m = offs_m < M
    valid_n = offs_n < N

    # 3. Dtype : Accumulator Must FP32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 4. ptr to Int64 (out of range in INT32) 
    x_ptr_base = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm
    
    bn      = offs_n.to(tl.int64) * stride_bn
    an      = offs_n.to(tl.int64) * stride_an
    bn_bias = offs_n.to(tl.int64) * stride_bn_bias

    shift_vals = tl.arange(0, 32)[:, None]
    k_offsets  = tl.arange(0, 32)[None, :]

    # --- Main Loop ---
    for k_block_start in range(0, K, BLOCK_K):
        current_group_idx = k_block_start // GROUP_SIZE
        
        bias_offset = current_group_idx * stride_bg + bn_bias
        bias_val = tl.load(bias_ptr + bias_offset, mask=valid_n, other=0.0).to(tl.float16)

        if N_PLANES >= 1:
            a0 = tl.load(alpha_ptr + current_group_idx * stride_ag + 0 * stride_ab + an, mask=valid_n, other=0.0).to(tl.float16)
        if N_PLANES >= 2:
            a1 = tl.load(alpha_ptr + current_group_idx * stride_ag + 1 * stride_ab + an, mask=valid_n, other=0.0).to(tl.float16)
        if N_PLANES >= 3:
            a2 = tl.load(alpha_ptr + current_group_idx * stride_ag + 2 * stride_ab + an, mask=valid_n, other=0.0).to(tl.float16)
        if N_PLANES >= 4:
            a3 = tl.load(alpha_ptr + current_group_idx * stride_ag + 3 * stride_ab + an, mask=valid_n, other=0.0).to(tl.float16)

        k_tile_base = k_block_start // 32

        # Loop over 32-column tiles inside the group
        for t in tl.static_range(0, BLOCK_K // 32):
            k_start = k_block_start + t * 32
            k_tile  = k_tile_base + t

            k_range = (k_start + k_offsets).to(tl.int64)
            x_ptrs  = x_ptr_base + k_range * stride_xk
            x_chunk = tl.load(x_ptrs, mask=valid_m[:, None], other=0.0).to(tl.bfloat16)

            w = tl.zeros((32, BLOCK_N), dtype=tl.float16)

            if N_PLANES >= 1:
                p0 = tl.load(B_ptr + k_tile * stride_bk + 0 * stride_bb + bn, mask=valid_n, other=0)
                b0 = ((p0[None, :] >> shift_vals) & 1).to(tl.float16)
                w += b0 * a0[None, :]
            
            if N_PLANES >= 2:
                p1 = tl.load(B_ptr + k_tile * stride_bk + 1 * stride_bb + bn, mask=valid_n, other=0)
                b1 = ((p1[None, :] >> shift_vals) & 1).to(tl.float16)
                w += b1 * a1[None, :]

            if N_PLANES >= 3:
                p2 = tl.load(B_ptr + k_tile * stride_bk + 2 * stride_bb + bn, mask=valid_n, other=0)
                b2 = ((p2[None, :] >> shift_vals) & 1).to(tl.float16)
                w += b2 * a2[None, :]

            if N_PLANES >= 4:
                p3 = tl.load(B_ptr + k_tile * stride_bk + 3 * stride_bb + bn, mask=valid_n, other=0)
                b3 = ((p3[None, :] >> shift_vals) & 1).to(tl.float16)
                w += b3 * a3[None, :]

            w_final = w + bias_val[None, :]

            # Dot Product (BF16 input -> FP32 acc)
            accumulator += tl.dot(x_chunk, w_final.to(tl.bfloat16))

    # Linear Bias
    if HAS_LINEAR_BIAS:
        out_b = tl.load(Linear_bias_ptr + offs_n.to(tl.int64) * stride_ob, mask=valid_n, other=0.0).to(tl.float32)
        accumulator += out_b[None, :]

    # Store
    out_ptrs = out_ptr + (offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :].to(tl.int64) * stride_on)
    tl.store(out_ptrs, accumulator.to(tl.bfloat16), mask=valid_m[:, None] & valid_n[None, :])




class BPDQTorchLinear(BaseQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.TORCH]
    SUPPORTS_METHODS = [METHOD.BPDQ]      
    SUPPORTS_FORMATS = {FORMAT.BPDQ: 10}  
    
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_GROUP_SIZE = [32, 64, 128, 256]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]    
    SUPPORTS_ADAPTERS = [] 
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]
    
    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "bpdq"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        self.msbits = kwargs.get("msbits", 4)
        
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TORCH),
            adapter=adapter,
            register_buffers=False,  
            **kwargs
        )

        if register_buffers:
            k = self.msbits
            num_column_groups = math.ceil(self.in_features / self.group_size)
            
            c_shape = (num_column_groups, k + 1, self.out_features)
            
            in_feat_padded = self.in_features
            if self.group_size == 256 and (in_feat_padded % 256 != 0):
                in_feat_padded = ((in_feat_padded + 256 - 1) // 256) * 256 
            b_shape = (in_feat_padded // 32, k, self.out_features)

            self.register_buffer("c", torch.zeros(c_shape, dtype=torch.float16))
            self.register_buffer("B", torch.zeros(b_shape, dtype=torch.int32))
            
            if bias:
                self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))
            else:
                self.bias = None

    def list_buffers(self) -> List:
        buf = []
        if hasattr(self, "c") and self.c is not None:
            buf.append(self.c)
        if hasattr(self, "B") and self.B is not None:
            buf.append(self.B)
        if hasattr(self, "bias") and self.bias is not None:
            buf.append(self.bias)
        return buf

    def pack(self, linear: nn.Module, c: torch.Tensor = None, B: torch.Tensor = None, **kwargs):
        if c is not None and B is not None:
            self.register_buffer("c", c)
            self.register_buffer("B", B)

        if getattr(linear, "bias", None) is not None:
            b = linear.bias.detach().to(device="cpu", dtype=torch.float16).contiguous()
            if isinstance(getattr(self, "_buffers", None), dict) and isinstance(self._buffers.get("bias", None), torch.Tensor):
                self._buffers["bias"].copy_(b)
            else:
                self.register_buffer("bias", b)
        else:
            if isinstance(getattr(self, "_buffers", None), dict):
                self._buffers.pop("bias", None)
            self.bias = None

    def _unpack_lutgemm_int32(self, packed_int32: torch.Tensor, original_k: int):
        K_tiles, Bits, M = packed_int32.shape
        
        shifts = torch.arange(32, device=packed_int32.device, dtype=torch.int32).view(1, 1, 1, 32)
        
        packed_expanded = packed_int32.unsqueeze(-1)
        unpacked_bits = (packed_expanded >> shifts) & 1 # [K_tiles, Bits, M, 32]

        #    Target: [Bits, M, K_tiles, 32] -> [Bits, M, Total_K]
        unpacked_permuted = unpacked_bits.permute(1, 2, 0, 3)
        
        B_restored = unpacked_permuted.contiguous().reshape(Bits, M, K_tiles * 32)
        if B_restored.shape[2] > original_k:
            B_restored = B_restored[:, :, :original_k]
            
        return B_restored 

    def _forward_python_ref(self, x_in):
        K_eff = x_in.shape[1] 

        B_unpacked = self._unpack_lutgemm_int32(self.B, K_eff).to(x_in.dtype)  # [Bits, M, K_eff]

        alpha = self._lut_alpha.to(x_in.dtype)  # [Groups, Bits, M]
        bias  = self._lut_bias.to(x_in.dtype)   # [Groups, M]

        col_group = torch.arange(K_eff, device=x_in.device, dtype=torch.long) // self.group_size  # [K_eff]

        alpha_k = alpha.index_select(0, col_group).permute(1, 2, 0).contiguous()  # [Bits, M, K_eff]
        bias_k  = bias.index_select(0, col_group).t().contiguous()                # [M, K_eff]

        W_recon = (B_unpacked.float() * alpha_k.float()).sum(dim=0) + bias_k.float()  # [M, K_eff]

        out = x_in.float().matmul(W_recon.t()).to(x_in.dtype)
        # return out

        if getattr(self, "bias", None) is not None:
            out = out + self.bias.to(device=out.device, dtype=out.dtype) 
        return out.to(x_in.dtype)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])

        ref_flag = False 

        if hasattr(self, "B") and self.B.dtype == torch.int32:
            input_dtype = x.dtype
            if input_dtype != torch.bfloat16:
                x_in = x.to(torch.bfloat16).contiguous()
            else:
                x_in = x.contiguous()


            if not hasattr(self, "_lut_cache_ready"):
                self._lut_alpha = self.c[:, :-1, :].to(torch.bfloat16).contiguous()
                self._lut_bias  = self.c[:, -1,  :].to(torch.bfloat16).contiguous()
                raw_bias = getattr(self, "bias", None)
                if raw_bias is not None:
                    self._cached_linear_bias = raw_bias.to(torch.bfloat16).contiguous()
                    self._cached_ob_stride   = self._cached_linear_bias.stride(0)
                else:
                    self._cached_linear_bias = None
                    self._cached_ob_stride   = 0
                self._lut_cache_ready = True


            total_tokens = x_in.shape[0]
               
            M, K = x_in.shape
            _, _, N = self.B.shape

            assert K % self.group_size == 0, f"K={K} must be divisible by group_size={self.group_size}"
            assert K % 32 == 0, "K must be divisible by 32 (bit-tile requirement)"

            out = torch.empty((M, N), device=x_in.device, dtype=torch.bfloat16)

            has_linear_bias = (self._cached_linear_bias is not None)
            stride_ob       = self._cached_ob_stride
            linear_bias = self._cached_linear_bias if has_linear_bias else out


            grid = lambda META: (
                triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
            )

            bpdq_kernel_v2[grid](
                # Pointers
                x_in, self.B, self._lut_alpha, self._lut_bias, out, linear_bias,
                # Shapes
                M, N, K,
                # Strides
                x_in.stride(0), x_in.stride(1),
                self.B.stride(0), self.B.stride(1), self.B.stride(2),
                self._lut_alpha.stride(0), self._lut_alpha.stride(1), self._lut_alpha.stride(2),
                self._lut_bias.stride(0), self._lut_bias.stride(1),
                out.stride(0), out.stride(1), stride_ob,
                BLOCK_K=self.group_size,   
                GROUP_SIZE=self.group_size,
                N_PLANES=self.msbits,
                HAS_LINEAR_BIAS=has_linear_bias,
            )

            if ref_flag and hasattr(self, "_forward_python_ref"):
                name = getattr(self, "name", self.__class__.__name__)
                out_ref = self._forward_python_ref(x_in)
                out_f = out.float()
                ref_f = out_ref.float()
                diff = (out_f - ref_f).abs()

                same_inf = torch.isinf(out_f) & torch.isinf(ref_f) & (torch.sign(out_f) == torch.sign(ref_f))
                diff = diff.masked_fill(same_inf, 0.0)

                nan_in  = torch.isnan(x_in).any().item()
                inf_in  = torch.isinf(x_in).any().item()
                nan_out = torch.isnan(out).any().item()
                inf_out = torch.isinf(out).any().item()
                nan_ref = torch.isnan(out_ref).any().item()
                inf_ref = torch.isinf(out_ref).any().item()

                
                total_tokens = x_in.shape[0]

                if nan_in or inf_in or nan_out or inf_out or nan_ref or inf_ref:
                    print(
                        f"[{name}] NON-FINITE "
                        f"in(nan={nan_in},inf={inf_in}) "
                        f"out(nan={nan_out},inf={inf_out}) "
                        f"ref(nan={nan_ref},inf={inf_ref}) | "
                        f"max|in|={x_in.abs().max().item():.3e} "
                        f"max|out|={out.abs().max().item():.3e} "
                        f"max|ref|={out_ref.abs().max().item():.3e}"
                    )

                # nan-safe mean/max
                diff_f = diff.float()
                mask = ~torch.isnan(diff_f)
                if mask.any():
                    mean_diff = diff_f[mask].mean().item()
                    max_diff  = diff_f[mask].max().item()
                else:
                    mean_diff = float("nan")
                    max_diff  = float("nan")

                status = "✅ MATCH" if (mean_diff == mean_diff and mean_diff < 1e-2) else "⚠️  MISMATCH"
                print(f"[{name}] tk={total_tokens} | MeanDiff: {mean_diff:.5f} MaxDiff: {max_diff:.5f} {status}")

            if input_dtype != torch.bfloat16:
                out = out.to(input_dtype)   

        else:
            out = self._forward(x, out_shape)

        return out.reshape(out_shape)
    

