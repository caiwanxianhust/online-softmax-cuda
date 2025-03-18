namespace softmax
{
    void launchSoftmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol, const int nrow, cudaStream_t stream = 0);
} // namespace softmax
