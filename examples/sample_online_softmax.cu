#include "utils.h"
#include "online_softmax.cuh"

#include <assert.h>
#include <cstdio>
// #include <string>

void printMatrix(const float *mat, char *s, int height, int width,
                 int end_row, int end_col, int start_row = 0, int start_col = 0)
{
    assert(start_row >= 0 && start_col >= 0 && end_row <= height && end_col <= width);
    printf("\nmatrix %s: width=%d, height=%d, start_row=%d, end_row=%d, start_col=%d, end_col=%d\n",
           s, width, height, start_row, end_row, start_col, end_col);
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = start_col; j < end_col; j++)
        {
            printf("%g\t", mat[i * width + j]);
        }
        printf("\n");
    }
}

void printVec(const float *vec, char *s, int length, int end_id, int start_id = 0)
{
    assert(start_id >= 0 && end_id <= length);
    printf("\nvec %s: length=%d, start_id=%d, end_id=%d\n", s, length, start_id, end_id);
    for (int i = start_id; i < end_id; i++)
    {
        printf("%g\t", vec[i]);
    }
    printf("\n");
}

void timingGemv(const float *A, float *B, const int M, const int N)
{
    constexpr int REPEAT_NUM = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < REPEAT_NUM; ++i)
    {
        softmax::launchOnlineSoftmaxKernel(A, B, N, M);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("alogrithm: online softmax kernel, elapsed_time: %g ms\n", elapsed_time / REPEAT_NUM);
}

int main(int argc, char *argv[])
{
    const int M = 128;
    const int N = 2048 * 128;

    float *h_a = new float[M * N];
    float *h_b = new float[M * N];

    for (int i = 0; i < M; ++i)
    {
        for (int j=0; j<N; ++j) {
            h_a[i * N + j] = (j == i) ? (float)i + 1.0f : 0.1f;
        }
    }

    printMatrix(h_a, (char *)("Matrix input: "), M, N, 128, 128, 112, 112);

    float *d_a;
    float *d_b;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_a, sizeof(float) * M * N));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_b, sizeof(float) * M * N));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, sizeof(float) * M * N, cudaMemcpyHostToDevice));

    timingGemv(d_a, d_b, M, N);

    CHECK_CUDA_ERROR(cudaMemcpy(h_b, d_b, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    printMatrix(h_b, (char *)("Matrix output: "), M, N, 128, 128, 112, 112);

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    delete [] h_a;
    delete [] h_b;

    return 0;
}
