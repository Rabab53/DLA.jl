using LinearAlgebra
using KernelAbstractions
using CUDA

"""
This function implements a GPU-accelerated algorithm to solve a system of linear equations of the form:
    A * x = b, where A is a lower triangular matrix.

It uses a two-step process to solve the system:
1. **Normalization**: For each row `i`, we divide both the corresponding element of `b` (i.e., `b[i]`) and the lower triangular elements of matrix `A` by the diagonal element `A[i, i]`.
2. **Substitution**: After normalization, we update each element of `b` by subtracting the contributions from the previously solved variables, based on the elements of `A`.

The solution is computed in-place, modifying the vector `b` to become the solution `x`.

This is implemented using a GPU kernel for parallel processing to speed up the computation, especially for large matrices.

The algorithm assumes:
- The matrix `A` is lower triangular.
- The system is not transposed, and the matrix is not upper triangular.

The implementation is optimized for GPU usage through CUDA and KernelAbstractions.
"""
@kernel function both_steps(A, B, n)
    # Allocate shared memory for diagonal and vector B
    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024

    # Determine the row index of the current thread
    row = @index(Global)

    if row <= n
        # Load the diagonal element and normalize B[row]
        diag[row] = A[row, row]
        B_c[row] = B[row] / diag[row]
    end

    # Substitution step
    for col in 1:n
        @synchronize
        if row > col
            B_c[row] -= (A[row, col] / diag[row]) * B_c[col]
        end
    end

    # Write back the result to B
    B[row, 1] = B_c[row]
end

function performant_trsm_2_2!(
    side::Char, uplo::Char, transposed::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where T
    if side == 'L' && uplo == 'L' && transposed == 'N'
        n = size(A, 1)

        # Get the backend (GPU)
        backend = get_backend(A)

        # Launch the kernel with GPU backend
        both_steps(backend, n)(A, B, n, ndrange=n)

        return B
    else
        error("Only the case where side='L', uplo='L', and transposed='N' is supported.")
    end
end