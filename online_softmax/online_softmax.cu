#include "online_softmax.cuh"
#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cub/cub.cuh>

namespace softmax
{
    namespace {
        constexpr int BLOCK_SIZE = 256;
    }
    template <typename T>
    struct MaxOp
    {
        __device__ __forceinline__ T operator()(const T &a, const T &b) { return max(a, b); }
    };

    template <typename T>
    struct SumOp
    {
        __device__ __forceinline__ T operator()(const T &a, const T &b) { return a + b; }
    };

    template <template <typename> class ReduceOp, typename T>
    __device__ __inline__ T warpAllReduce(T val)
    {
        auto functor = ReduceOp<T>();
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            val = functor(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
        }
        return val;
    }

    template <typename T>
    __device__ __inline__ T blockAllReduceSum(T val)
    {
        __shared__ T shared[32];
        __shared__ T ret;
        int warp_id = (threadIdx.x >> 5);
        int lane_id = (threadIdx.x & 31);

        val = warpAllReduce<SumOp, T>(val);
        if (lane_id == 0)
        {
            shared[warp_id] = val;
        }
        __syncthreads();

        val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : (T)(0.0f);
        val = warpAllReduce<SumOp, T>(val);
        if (threadIdx.x == 0)
        {
            ret = val;
        }
        __syncthreads();

        return ret;
    }

    template <typename T>
    __device__ __inline__ T blockAllReduceMax(T val)
    {
        __shared__ T shared[32];
        __shared__ T ret;
        int warp_id = (threadIdx.x >> 5);
        int lane_id = (threadIdx.x & 31);

        val = warpAllReduce<MaxOp, T>(val);
        if (lane_id == 0)
        {
            shared[warp_id] = val;
        }
        __syncthreads();

        val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : (T)(-FLT_MAX);
        val = warpAllReduce<MaxOp, T>(val);
        if (threadIdx.x == 0)
        {
            ret = val;
        }
        __syncthreads();

        return ret;
    }

    __global__ void softmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol)
    {
        float val;
        float vmax = -FLT_MAX;
        float exp_sum = 1e-10f;

#pragma unroll
        for (int i = threadIdx.x; i < ncol; i += blockDim.x)
        {
            vmax = max(mat[blockIdx.x * ncol + i], vmax);
        }
        __syncthreads();

        vmax = blockAllReduceMax<float>(vmax);

#pragma unroll
        for (int i = threadIdx.x; i < ncol; i += blockDim.x)
        {
            exp_sum += __expf(mat[blockIdx.x * ncol + i] - vmax);
        }
        __syncthreads();

        exp_sum = blockAllReduceSum<float>(exp_sum);

#pragma unroll
        for (int i = threadIdx.x; i < ncol; i += blockDim.x)
        {
            val = __expf(mat[blockIdx.x * ncol + i] - vmax) / exp_sum;
            output[blockIdx.x * ncol + i] = val;
        }
    }

    void launchSoftmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol, const int nrow, cudaStream_t stream)
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid(nrow);
        softmaxKernel<<<grid, block, 0, stream>>>(mat, output, ncol);
    }

    struct __align__(8) MD_F
    {
        float m; // max val
        float d; // exp sum
    };

    __device__ __forceinline__ MD_F reduceMDFOp(MD_F &a, MD_F &b)
    {
        MD_F ret;
        ret.m = max(a.m, b.m);
        ret.d = a.d * __expf(a.m - ret.m) + b.d * __expf(b.m - ret.m);
        return ret;
    }

    __global__ void onlineSoftmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol)
    {
        MD_F mdf_tmp, mdf_val;
        mdf_val.d = 0.0f;
        mdf_val.m = -1e20f;

#pragma unroll
        for (int i = threadIdx.x; i < ncol; i += blockDim.x)
        {
            mdf_tmp.m = mat[blockIdx.x * ncol + i];
            mdf_tmp.d = 1.0f;
            mdf_val = reduceMDFOp(mdf_tmp, mdf_val);
        }
        __syncthreads();

        typedef cub::BlockReduce<MD_F, BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        __shared__ MD_F mdf_total;
        mdf_val = BlockReduce(tempStorage).Reduce(mdf_val, reduceMDFOp);
        if (threadIdx.x == 0)
        {
            mdf_total = mdf_val;
        }
        __syncthreads();

#pragma unroll
        for (int i = threadIdx.x; i < ncol; i += blockDim.x)
        {
            output[blockIdx.x * ncol + i] = __expf(mat[blockIdx.x * ncol + i] - mdf_total.m) / mdf_total.d;
        }
    }

    void launchOnlineSoftmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol, const int nrow, cudaStream_t stream)
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid(nrow);
        onlineSoftmaxKernel<<<grid, block, 0, stream>>>(mat, output, ncol);
    }

} // namespace softmax