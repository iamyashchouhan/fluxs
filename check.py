import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA driver
import numpy as np
from pycuda.compiler import SourceModule
import time

# Define a CUDA kernel to perform matrix multiplication
kernel_code = """
__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
"""

def stress_gpu():

    N = 15024


    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    C = np.zeros_like(A)


    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(C.nbytes)


    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)


    mod = SourceModule(kernel_code)
    matmul = mod.get_function("matmul")


    start_time = time.time()


    block_size = 32
    grid_size = (N // block_size, N // block_size)
    matmul(A_gpu, B_gpu, C_gpu, np.int32(N), block=(block_size, block_size, 1), grid=grid_size)


    cuda.memcpy_dtoh(C, C_gpu)


    end_time = time.time()


    print(f"GPU Stress Test Completed in {end_time - start_time:.2f} seconds.")
    print("Matrix multiplication complete.")

if __name__ == "__main__":
    stress_gpu()
