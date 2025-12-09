# from deep_gemm import quantize
import deep_gemm
import torch
import random
from deep_gemm.testing import (bench_kineto, count_bytes)
from deep_gemm.utils import (
    align,
    ceil_to_ue8m0
)


def per_token_cast_to_fp8(x: torch.Tensor, block_size: int, use_ue8m0: bool):
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, block_size)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, block_size)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous(), sf


def test_quant():
    # Only support k = 8192 for this kernel
    m, k, transposed = 131072, 8192, False

    a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')

    ref_fp8_a, ref_scale = per_token_cast_to_fp8(a, block_size=32, use_ue8m0=True)
    fp8_a, scale = deep_gemm.quantize_bf16_to_fp8(a)

    def test_func():
        fp8_a, scale = deep_gemm.quantize_bf16_to_fp8(a)

    assert(torch.equal(fp8_a, ref_fp8_a))

    t = bench_kineto(test_func, 'quantize_bf16_to_fp8', suppress_kineto_output=True)
    print(f'{count_bytes(a, fp8_a, scale) / 1e9 / t:4.0f} GB/s')


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    test_quant()
