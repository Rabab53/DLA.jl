using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc


"""
Computes a blocked QR factorization of a real or complex "triangular-pentagonal" matrix, which is 
composed of a triangular block and a pentagonal block, using the compact WY representation for Q.
"""
##TODO this for Float64 you have to put it in for loop to generate for all types see: vim ~/julia-1.9.3/share/julia/stdlib/v1.9/LinearAlgebra/src/lapack.jl
function tpqrt!(::Type{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, 
    Tau::AbstractMatrix{T}) where {T<: Number}
    
    Am, An = size(B)
    ib, nb = size(Tau)
    Bm, Bn = size(B)
    lda = max(1, stride(A,2))
    ldb = max(1, stride(B,2))
    work = Vector{T}(undef, (ib+1)*An)
    if An > 0
        info = Ref{BlasInt}()
        ccall((@blasfunc(dtpqrt_), libblastrampoline), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ptr{BlasInt}),
            Am, An, Bm, ib, A, lda, B, ldb, Tau, max(1,stride(Tau,2)), 
            work, info)
        chklapackerror(info[])
    end
end

A = rand(4, 4)
B = rand(4, 4)
Tau = rand(1, 4)
tpqrt!(Float64, A, B, Tau)