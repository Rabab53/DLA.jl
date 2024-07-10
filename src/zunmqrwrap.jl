using LinearAlgebra 
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc

# SUBROUTINE ZUNMQR( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC, WORK, LWORK, INFO )

# int coreblas_zunmqr(coreblas_enum_t side, coreblas_enum_t trans,
# int m, int n, int k, int ib,
# const coreblas_complex64_t *A,    int lda,
#const coreblas_complex64_t *T,    int ldt,
#  coreblas_complex64_t *C,    int ldc,
#     coreblas_complex64_t *work, int ldwork)
# coreblas_enum_t = int
#
function unmqr!(::Type{T}, side::AbstractChar, trans::AbstractChar, 
    A::AbstractMatrix{T}, Tau::AbstractVector{T}, 
    C::AbstractMatrix{T})  where {T<: Number}

    mA, k = size(A)
    m,n = size(C)
    # ib, nb = size(Tau)
    #tau has dimension (k)
    lda = max(1, stride(A,2))
    ldc = max(1, stride(C,2))
    work = Vector{T}(undef, 1)
    lwork = BlasInt(-1)
    info = Ref{BlasInt}()

    #tau2 = vec(Tau)

    for i in 1:2
        if T == ComplexF64
            ccall((@blasfunc(zunmqr_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}),
            side, trans, m, n, k, A, lda, Tau, 
            C, ldc, work, lwork, info)
    
            if i == 1
                lwork = BlasInt(real(work[1]))
                resize!(work, lwork)
            end
    
            chklapackerror(info[])
        elseif T == Float64
            ccall((@blasfunc(dormqr_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}),
            side, trans, m, n, k, A, lda, Tau, 
            C, ldc, work, lwork, info)
    
            if i == 1
                lwork = BlasInt(real(work[1]))
                resize!(work, lwork)
            end
    
            chklapackerror(info[])        
        elseif T == ComplexF32
            ccall((@blasfunc(cunmqr_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}),
            side, trans, m, n, k, A, lda, Tau, 
            C, ldc, work, lwork, info)
    
            if i == 1
                lwork = BlasInt(real(work[1]))
                resize!(work, lwork)
            end
    
            chklapackerror(info[])
        else #T = Float32
            ccall((@blasfunc(sormqr_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}),
            side, trans, m, n, k, A, lda, Tau, 
            C, ldc, work, lwork, info)
    
            if i == 1
                lwork = BlasInt(real(work[1]))
                resize!(work, lwork)
            end
    
            chklapackerror(info[])
        end

    end
end

"""
side = 'L'
trans = 'N'
ib = 2
m = n = 6
k = m
Tau = rand(ComplexF64,ib, k)
C = rand(ComplexF64, m,n)
A = rand(ComplexF64,m,k)

display(C)
unmqr!(ComplexF64, side, trans, A, Tau, C)
display(C)
"""