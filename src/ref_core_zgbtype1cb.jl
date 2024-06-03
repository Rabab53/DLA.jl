using LinearAlgebra, libcoreblas_jll, OpenBLAS32_jll

function ref_coreblas_zgbtype1cb!(::Type{T}, n, nb, A::AbstractMatrix{T}, 
    VQ::Vector{T},  TAUQ::Vector{T}, VP::Vector{T},  TAUP::Vector{T}) where {T<: Number}
    m1, n1 = size(A)
    nq = size(VQ)
    np = size(VP)
    uplo = 121
    nb = nb
    st=0
    ed=4
    sweep=1
    Vblksiz=1
    wantz=0
    work = Vector{T}(undef, nb)

    @show m1, n1

    ccall((:coreblas_dgbtype1cb, "libcoreblas.so"), Cvoid,
            (Int64, Int64, Int64, Ptr{T}, Int64, Ptr{T}, Ptr{T},
            Ptr{T}, Ptr{T}, Int64, Int64, Int64, Int64, Int64, Ptr{T}),
            uplo, n, nb, A, m1, VQ, TAUQ, VP, TAUP, st, ed, sweep, Vblksiz, wantz, work)
    
end
n=4
nb=2
A = rand(3*nb+1, n)
VP = zeros(n)
VQ = zeros(n)
TAUP = zeros(n)
TAUQ = zeros(n)
ref_coreblas_zgbtype1cb!(Float64, n, nb, A, VQ, TAUQ, VP, TAUP)
display(A)
display(VP)
display(VQ)
display(TAUP)
display(TAUQ)