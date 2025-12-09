#pragma once

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumThreads, uint32_t BLOCK_MN, uint32_t SF_K,
          uint32_t PADDED_SF_K = SF_K + (1 - (SF_K % 2))>
__global__ void transpose_fp32(const float* sf, float* out, const uint32_t mn) {
    typedef typename Vectorized<sizeof(float) * SF_K>::vec_t in_vec_t;
    constexpr static uint32_t kNumElemsPerVec = sizeof(in_vec_t) / sizeof(float);
    constexpr static uint32_t SF_VEC_K = SF_K / kNumElemsPerVec;

    // Shapes and strides
    extern __shared__ float smem_buffer[];
    constexpr auto kNumTMAAlignedElems = static_cast<uint32_t>(16 / sizeof(float));
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto tma_aligned_mn = align<uint32_t>(mn, kNumTMAAlignedElems);

    // Shift into the block
    sf = sf + static_cast<uint64_t>(blockIdx.y) * mn * SF_K;
    out = out + static_cast<uint64_t>(blockIdx.y) * tma_aligned_mn * SF_K;
    const auto& local_sf = reinterpret_cast<const in_vec_t*>(sf + static_cast<uint64_t>(blockIdx.x) * (BLOCK_MN * SF_K));

    // Load
    for (uint32_t i = threadIdx.x; i < in_block_mn * SF_VEC_K; i += kNumThreads) {
        auto in_vec = __ldg(local_sf + i);
        const auto& in_values = reinterpret_cast<float*>(&in_vec);

        const auto& row = i / SF_VEC_K, col = (i % SF_VEC_K) * kNumElemsPerVec;
        #pragma unroll
        for (uint32_t j = 0; j < kNumElemsPerVec; ++ j)
            smem_buffer[row * PADDED_SF_K + col + j] = in_values[j];
    }
    __syncthreads();

    // Store
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < in_block_mn * SF_K; i += kNumThreads) {
        const auto& sf_k_idx = i / in_block_mn, mn_idx = i % in_block_mn;
        const auto& global_mn_idx = blockIdx.x * BLOCK_MN + mn_idx;
        out[sf_k_idx * tma_aligned_mn + global_mn_idx] = ld_shared(smem_buffer + mn_idx * PADDED_SF_K + sf_k_idx);
    }
}

// NOTES: the two kernels below always pack the K dimension

template <uint32_t kNumThreads, uint32_t BLOCK_MN, uint32_t SF_K>
__global__ void transpose_and_pack_fp32_into_ue8m0(float* sf, uint32_t* out, const uint32_t mn) {
    extern __shared__ uint32_t smem_buffer[];

    // Shapes and strides
    constexpr auto kNumPackedSFK = constexpr_ceil_div(SF_K, 4u);
    constexpr auto kNumTMAAlignedElems = static_cast<uint32_t>(16 / sizeof(int));
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto tma_aligned_mn = align<uint64_t>(mn, kNumTMAAlignedElems);

    // Shift into the group
    sf = sf + static_cast<uint64_t>(blockIdx.y) * mn * SF_K;
    out = out + static_cast<uint64_t>(blockIdx.y) * tma_aligned_mn * kNumPackedSFK;

    // Load FP32 SFs
    DG_STATIC_ASSERT(BLOCK_MN % 4 == 0, "Invalid block size");
    const auto local_sf = reinterpret_cast<uint32_t*>(sf + static_cast<uint64_t>(blockIdx.x) * (BLOCK_MN * SF_K));
    const auto num_values = in_block_mn * SF_K;
    const auto num_uint4 = num_values / 4;
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < num_uint4; i += kNumThreads) {
        const auto& [x, y, z, w] = __ldg(reinterpret_cast<uint4*>(local_sf) + i);
        st_shared(reinterpret_cast<uint4*>(smem_buffer) + i, x, y, z, w);
    }

    // Fill unaligned values as well
    if (const auto unaligned_idx = num_uint4 * 4 + threadIdx.x; unaligned_idx < num_values)
        st_shared(smem_buffer + unaligned_idx, __ldg(local_sf + unaligned_idx));
    __syncthreads();

    // Pack into UE8M0 and store
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < (kNumPackedSFK * BLOCK_MN); i += kNumThreads) {
        const auto sf_k_pack_idx = i / BLOCK_MN, mn_idx = i % BLOCK_MN;

        // Load shared memory
        uint32_t values[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            const auto sf_k_idx = sf_k_pack_idx * 4 + j;
            values[j] = sf_k_idx < SF_K ? ld_shared(smem_buffer + mn_idx * SF_K + sf_k_idx) : 0;
        }

        // Pack and store
        uint32_t packed = 0;
        packed |= (values[0] >> 23u);
        packed |= (values[1] >> 15u);
        packed |= (values[2] >>  7u);
        packed |= (values[3] <<  1u);
        if (const auto global_mn_idx = blockIdx.x * BLOCK_MN + mn_idx; global_mn_idx < mn)
            out[sf_k_pack_idx * tma_aligned_mn + global_mn_idx] = packed;
    }
}

template <uint32_t kNumGroups, uint32_t kNumThreads,
          uint32_t BLOCK_MN, uint32_t BLOCK_PACKED_SF_K, bool kTransposed = true>
__global__ void pack_fp32_into_ue8m0(float* sf, uint32_t* out, uint32_t* ks,
                                     const uint32_t mn, uint32_t sf_k, const uint32_t packed_sf_k) {
    // Always packing the K dimension
    // NOTES: should also assert `mn % 4 == 0` at launch
    DG_STATIC_ASSERT(kTransposed, "Currently only support transposed SFs (MN-major)");
    DG_STATIC_ASSERT(BLOCK_MN % 4 == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_PACKED_SF_K == kNumThreads / 32, "Invalid block sizes");

    // Shapes and strides
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto in_block_mn_uint4 = in_block_mn / 4;
    const auto in_block_packed_sf_k = min(BLOCK_PACKED_SF_K, packed_sf_k - blockIdx.y * BLOCK_PACKED_SF_K);

    // Shift into the right block along MN
    sf += blockIdx.x * BLOCK_MN;
    out += blockIdx.x * BLOCK_MN;

    // Each warp is responsible for a packed row
    const auto warp_idx = threadIdx.x / 32;
    const auto lane_idx = get_lane_idx();
    const auto packed_sf_k_idx = static_cast<uint64_t>(blockIdx.y) * BLOCK_PACKED_SF_K + warp_idx;
    if (warp_idx >= in_block_packed_sf_k)
        return;

    // Make an offset on the input
    uint32_t input_offset = 0;
    if constexpr (kNumGroups > 1) {
        // Load each group's size
        DG_STATIC_ASSERT(kNumGroups <= 128, "Too many groups");
        uint32_t group_ks[4];
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++ i) {
            const auto group_idx = lane_idx * 4 + i;
            group_ks[i] = group_idx < kNumGroups ? __ldg(ks + group_idx) : 0;
        }
        __syncwarp();

        // Make the offset
        sf_k = 0;
        auto sum_packed_sf_k = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumGroups; ++ i) {
            const auto sf_k_in_group = __shfl_sync(0xffffffff, group_ks[i % 4] / 128, i / 4);
            sf_k += sf_k_in_group;
            sum_packed_sf_k += ceil_div(sf_k_in_group, 4u);
            if (packed_sf_k_idx < sum_packed_sf_k)
                break;
            if (const auto remainder = sf_k_in_group % 4; remainder > 0)
                input_offset += 4 - remainder;
        }
    }

    for (uint32_t mn_idx = get_lane_idx(); mn_idx < in_block_mn_uint4; mn_idx += 32) {
        // Load
        uint4 values[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            values[j] = make_uint4(0, 0, 0, 0);
            if (const auto sf_k_idx = packed_sf_k_idx * 4 + j - input_offset; sf_k_idx < sf_k)
                values[j] = __ldg(reinterpret_cast<uint4*>(sf + sf_k_idx * mn) + mn_idx);
        }

        // Pack and store
        uint4 packed;
        packed.x = (values[0].x >> 23u) | (values[1].x >> 15u) | (values[2].x >> 7u) | (values[3].x << 1u);
        packed.y = (values[0].y >> 23u) | (values[1].y >> 15u) | (values[2].y >> 7u) | (values[3].y << 1u);
        packed.z = (values[0].z >> 23u) | (values[1].z >> 15u) | (values[2].z >> 7u) | (values[3].z << 1u);
        packed.w = (values[0].w >> 23u) | (values[1].w >> 15u) | (values[2].w >> 7u) | (values[3].w << 1u);
        reinterpret_cast<uint4*>(out + packed_sf_k_idx * mn)[mn_idx] = packed;
    }
}

using e8m0_t = uint8_t;
using bfloat16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;

// FP32 constants
constexpr int32_t FP32_MANTISSA_BITS = 23;
constexpr int32_t FP32_EXPONENT_BIAS = 127;

// BF16 constants
constexpr int32_t BF16_MANTISSA_BITS = 7;
constexpr int32_t BF16_EXPONENT_BIAS = 127;

// FP8E4M3 constants
constexpr int32_t F8E4M3_MAX_POW2 = 8;
constexpr float F8E4M3_MAX = 448.0;

// FP8E8M0 constants
constexpr int32_t E8M0_EXPONENT_BIAS = 127;

__device__ __forceinline__
uint16_t float2_to_e4m3x2(float2 x) {
    uint16_t out;
    // x.x -> 低 8 bit，x.y -> 高 8 bit
    asm volatile(
        "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
        : "=h"(out)
        : "f"(x.y), "f"(x.x));
    return out;
}

// Source:
// https://github.com/NVIDIA/TransformerEngine/blob/1ae1d228d725a488621deba685bd26d6ee1cdb21/transformer_engine/common/utils.cuh#L937
__device__ __forceinline__ e8m0_t float_to_e8m0(float val) {
    // TODO: nan/inf needs to be set for any value
    // of nan/inf in input not just amax.
    if (isnan(val)) {
      return 0xFF;
    }
    if (isinf(val)) {
      return 0xFE;
    }
  #if ((__CUDA_ARCH_HAS_FEATURE__(SM100_ALL)) ||                                 \
       (__CUDA_ARCH_HAS_FEATURE__(SM101_ALL)) ||                                 \
       (__CUDA_ARCH_HAS_FEATURE__(SM120_ALL)))
    uint16_t out;
    asm volatile("{\n"
                 "cvt.rp.satfinite.ue8m0x2.f32  %0, 0.0, %1;\n"
                 "}"
                 : "=h"(out)
                 : "f"(val));
    return *reinterpret_cast<e8m0_t *>(&out);
  #else
    if (val == 0.0f) {
      return 0x00;
    }
    uint32_t val_u32 = *reinterpret_cast<uint32_t *>(&val);
    e8m0_t exponent = (val_u32 >> FP32_MANTISSA_BITS);
    uint32_t mantissa = val_u32 & 0x7FFFFF;
    // Round up exponent and deal with satfinite.
    if ((mantissa > 0 && exponent != 0xFE) &&
        !(exponent == 0 && mantissa <= 0x400000)) {
      ++exponent;
    }
    return exponent;
  #endif
}

// Source:
// https://github.com/NVIDIA/TransformerEngine/blob/1ae1d228d725a488621deba685bd26d6ee1cdb21/transformer_engine/common/utils.cuh#L971
__device__ __forceinline__ float exp2f_rcp(e8m0_t biased_exp) {
  return (biased_exp == 0)
             ? 1
             : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(biased_exp));
}

__device__ __forceinline__ uint4 ldg_uint4_ptx(const uint4* addr) {
    uint4 v;
    asm volatile(
        "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(addr));
    return v;
}

struct float8 {
    float x0, x1, x2, x3;
    float x4, x5, x6, x7;
};

__device__ __forceinline__ float8 ldg_256B(const float* addr) {
    float8 v;
    asm volatile(
        "ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
        : "=f"(v.x0), "=f"(v.x1), "=f"(v.x2), "=f"(v.x3),
          "=f"(v.x4), "=f"(v.x5), "=f"(v.x6), "=f"(v.x7)
        : "l"(addr));
    return v;
}

template <uint32_t SF_BLOCK_SIZE>
__global__ __launch_bounds__(256, 1) void quantize_bf16_to_fp8_kernel(const __nv_bfloat16* in, __nv_fp8_e4m3* out, uint32_t* sf_out, size_t num_rows, size_t num_cols) {
    size_t row_idx = blockIdx.x;
    size_t col_idx = threadIdx.x * 4;

    const float* row_ptr =  reinterpret_cast<const float*>(in + row_idx * num_cols);
    uint2* out_row_ptr = reinterpret_cast<uint2*>(out + row_idx * num_cols);

    float8 value[2];
    value[0] = ldg_256B(row_ptr + col_idx * 4);
    value[1] = ldg_256B(row_ptr + col_idx * 4 + 8);

    uint2 fp8_value[4];
    float amax = 0;

    __nv_bfloat16* bf16_ptr = reinterpret_cast<__nv_bfloat16*>(&value[0]);
    __nv_fp8_e4m3* fp8_value_ptr = reinterpret_cast<__nv_fp8_e4m3*>(&fp8_value[0]);

    uint16_t* e4m3x2_ptr = reinterpret_cast<uint16_t*>(fp8_value_ptr);

    for (int i = 0; i < 32; ++i) {
        float fp32_value = bf16_ptr[i];
        amax = max(amax, fabs(fp32_value));
    }

    float scale = amax / 448.0;

    float inv_scale_fp32;
    auto out_scale = float_to_e8m0(amax * (1.0f / 448.0f));
    inv_scale_fp32 = exp2f_rcp(out_scale);

    for (int i = 0; i < 16; ++i) {
        float fp32_value_0 = bf16_ptr[i * 2];
        float fp32_value_1 = bf16_ptr[i * 2 + 1];
        float2 x = make_float2(fp32_value_0, fp32_value_1);
        float2 scale = make_float2(inv_scale_fp32, inv_scale_fp32);
        float2 x_scaled = __fmul2_rn(x, scale);
        e4m3x2_ptr[i] = float2_to_e4m3x2(x_scaled);
    }

    out_row_ptr[col_idx] = fp8_value[0];
    out_row_ptr[col_idx + 1] = fp8_value[1];
    out_row_ptr[col_idx + 2] = fp8_value[2];
    out_row_ptr[col_idx + 3] = fp8_value[3];
}

} // namespace deep_gemm
