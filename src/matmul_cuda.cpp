#include <torch/extension.h>

// CUDA forward declarations
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_shared_memory_cuda(torch::Tensor a, torch::Tensor b, const int block_size = 16);
torch::Tensor matmul_shared_memory_4x4reg_cuda(torch::Tensor a, torch::Tensor b, const int block_size = 16);


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    return matmul_cuda(a, b);
}

torch::Tensor matmul_shared_memory(torch::Tensor a, torch::Tensor b, const int block_size = 16) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    return matmul_shared_memory_cuda(a, b, block_size);
}

torch::Tensor matmul_shared_memory_4x4reg(torch::Tensor a, torch::Tensor b, const int block_size = 16) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    return matmul_shared_memory_4x4reg_cuda(a, b, block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "MatMul implemented by CUDA.");
    m.def("matmul_shared_memory", &matmul_shared_memory, "MatMul implemented by CUDA with shared memory.");
    m.def("matmul_shared_memory_4x4reg", &matmul_shared_memory_4x4reg, "MatMul implemented by CUDA with shared memory and register optimal.");
}