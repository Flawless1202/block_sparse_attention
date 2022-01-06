#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <exception>

template <typename scalar_t>
__global__ void matmul_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> out,
    const int num_heads,
    const int m,
    const int n,
    const int k) {
    const int bidx = blockIdx.z;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        for (int hidx = 0; hidx < num_heads; hidx++) {
            scalar_t val = 0.0;
            scalar_t y = 0.0;
            for (int i = 0; i < k; i++) {
                val += a[bidx][hidx][row][i] * b[bidx][hidx][i][col];
                // y -= a[bidx][hidx][row][i] * b[bidx][hidx][i][col];
                // scalar_t r = val - y;
                // y = (r - val) + y;
                // val = r;
            }
            out[bidx][hidx][row][col] = val;
        }
    }
}

template <typename scalar_t>
__global__ void matmul_shared_memory_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> out,
    const int num_heads,
    const int m,
    const int n,
    const int k,
    const int block_size) {
    const int bidx = blockIdx.z;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    // __shared__ scalar_t _sh_tile_a[block_size][block_size];
    // __shared__ scalar_t _sh_tile_b[block_size][block_size];

    extern __shared__ char data[];
    scalar_t* _sh_tile_a = (scalar_t*)data;
    scalar_t* _sh_tile_b = (scalar_t*)(block_size * block_size * sizeof(scalar_t) + data);

    const int num_blocks = (k + block_size - 1) / block_size;

    if (row < m && col < n) {
        for (int hidx = 0; hidx < num_heads; hidx++) {
            scalar_t val = 0.0;
            for (int i = 0; i < num_blocks; i++) {
                _sh_tile_a[threadIdx.x * block_size + threadIdx.y] = a[bidx][hidx][row][i * block_size + threadIdx.y];
                _sh_tile_b[threadIdx.x * block_size + threadIdx.y] = b[bidx][hidx][i * block_size + threadIdx.x][col];
                __syncthreads();

                for (int j = 0; j < block_size; j++) {
                    val += _sh_tile_a[threadIdx.x * block_size + j] * _sh_tile_b[j * block_size + threadIdx.y];
                }
                __syncthreads();
            }
            out[bidx][hidx][row][col] = val;
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int bs = a.size(0);
    const int h = a.size(1);

    // the tensor a is of size `(bs, h, ma, ka)`
    const int ma = a.size(-2);
    const int ka = a.size(-1);

    // the tensor b is of size `(bs, h, kb, nb)`
    const int kb = b.size(-2);
    const int nb = b.size(-1);

    if (ka != kb) {
        throw std::invalid_argument("Size of tensor A must match size of tensor B.");
    }

    // configure cuda
    const int threads = 32;
    const dim3 threads_per_block(threads, threads, 1);
    const dim3 blocks_per_grid(ma / threads + 1, nb / threads + 1, bs);

    auto tensor_options = torch::TensorOptions().dtype(a.dtype()).device(torch::kCUDA, a.device().index());
    auto out = torch::zeros({bs, h, ma, nb}, tensor_options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        a.type(), "matmul_cuda", ([&] {
            matmul_cuda_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                a.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                b.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), h, ma, nb, ka);
        }));
    return out;
}

torch::Tensor matmul_shared_memory_cuda(torch::Tensor a, torch::Tensor b, const int block_size = 16) {
    const int bs = a.size(0);
    const int h = a.size(1);

    // the tensor a is of size `(bs, h, ma, ka)`
    const int ma = a.size(-2);
    const int ka = a.size(-1);

    // the tensor b is of size `(bs, h, kb, nb)`
    const int kb = b.size(-2);
    const int nb = b.size(-1);

    if (ka != kb) {
        throw std::invalid_argument("Size of tensor A must match size of tensor B.");
    }

    // configure cuda
    const dim3 threads_per_block(block_size, block_size, 1);
    const dim3 blocks_per_grid(ma / block_size + 1, nb / block_size + 1, bs);

    auto tensor_options = torch::TensorOptions().dtype(a.dtype()).device(torch::kCUDA, a.device().index());
    auto out = torch::zeros({bs, h, ma, nb}, tensor_options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        a.type(), "matmul_shared_memory_cuda", ([&] {
            matmul_shared_memory_cuda_kernel<scalar_t>
                <<<blocks_per_grid, threads_per_block, 2 * block_size * block_size * sizeof(scalar_t)>>>(
                    a.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    b.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), h, ma, nb, ka, block_size);
        }));
    return out;
}
