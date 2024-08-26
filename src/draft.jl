using Revise
using LinearAlgebra
#using DLA

import LinearAlgebra: PivotingStrategy

struct CompletePivoting <: PivotingStrategy end

for (getc2,T) in
    ((:dgetc2_, Float64),
     (:sgetc2_, Float32),
     (:zgetc2_, ComplexF64), 
     (:cgetc2_, ComplexF32))
    @eval begin
        function getc2(A::AbstractMatrix{$T}, ipiv::AbstractVector{BLAS.BlasInt}, jpiv::AbstractVector{BLAS.BlasInt}; check::Bool=true)
            LinearAlgebra.require_one_based_indexing(A)
            check && LAPACK.chkfinite(A)
            m, n = size(A)
            lda = max(1, stride(A, 2))
            info = Ref{BLAS.BlasInt}()
            ccall((BLAS.@blasfunc($getc2), LinearAlgebra.libblastrampoline), Cvoid,
                  (Ref{BLAS.BlasInt}, Ptr{$T}, Ref{BLAS.BlasInt}, Ptr{BLAS.BlasInt}, Ptr{BLAS.BlasInt}, Ref{BLAS.BlasInt}),
                  n, A, lda, ipiv, jpiv, info)
            return info[]
        end
    end
end

struct LU{T,S<:AbstractMatrix{T},P<:AbstractVector{<:Integer}} 
    factors::S
    ipiv::P
    jpiv::P
    info::LinearAlgebra.BlasInt # Can be negative to indicate failed unpivoted factorization

    function LU{T,S,P}(factors, ipiv, jpiv, info) where {T, S<:AbstractMatrix{T}, P<:AbstractVector{<:Integer}}
        new{T,S,P}(factors, ipiv, jpiv, info)
    end
end

function lu!(A::AbstractMatrix{T}, ::CompletePivoting; check::Bool = true, allowsingular::Bool = false) where {T<:LinearAlgebra.BlasFloat}
    m, n = size(A)
    if m == n && A isa StridedMatrix{T}
        ipiv = Vector{Int}(undef, m)
        jpiv = Vector{Int}(undef, m)
        info = getc2(A, ipiv, jpiv; check)
        check && _check_lu_success(info, allowsingular)
        return LU{T,typeof(A),typeof(ipiv)}(A, ipiv, jpiv, convert(LinearAlgebra.BlasInt, info))
    else
        generic_lufact!(A, CompletePivoting(), check, allowsingular)
    end
end

checknozeropivot(info) = info == 0 || throw(LinearAlgebra.ZeroPivotException(info))

checknonsingular(info) = info == 0 || throw(LinearAlgebra.SingularException(info))

function _check_lu_success(info, allowsingular)
    if info < 0
        checknozeropivot(-info)
    else
        allowsingular || checknonsingular(info)
    end
end

function generic_lufact!(A::AbstractMatrix{T}, pivot::Union{CompletePivoting, RowMaximum, RowNonZero, NoPivot} = lupivottype(T); check::Bool = true, allowsingular::Bool = false) where {T}
    check && LAPACK.chkfinite(A)
    # Extract values

    m, n = size(A)
    minmn = min(m,n)

    # Initialize variables
    info = 0
    ipiv = Vector{Int}(undef, minmn)
    jpiv = Vector{Int}(undef, minmn)
    @inbounds begin
        for k = 1:minmn
            # Find maximum element in the submatrix A[k:m, k:n]
            kp = k
            jp = k
            if pivot === RowMaximum() && k < m
                amax = abs(A[k, k])
                for i = k+1:m
                    absi = abs(A[i,k])
                    if absi > amax
                        kp = i 
                        amax = absi
                    end
                end
            elseif pivot === RowNonZero()
                for i = k:m
                    if !iszero(A[i,k])
                        kp = i
                        break
                    end
                end
            elseif pivot === CompletePivoting()
                amax = abs(A[k, k])
                for i = k+1:m
                    for j = k+1:n
                        absi = abs(A[i,j])
                        if absi > amax
                            kp = i
                            jp = j
                            amax = absi
                        end
                    end
                end
            end

            ipiv[k] = kp
            jpiv[k] = jp
            if !iszero(A[kp,jp])
                if k != kp
                    # Interchange rows
                    A[k,:], A[kp,:] = A[kp,:], A[k,:]
                end
                if k != jp
                    # Interchange columns
                    A[:,k], A[:,jp] = A[:,jp], A[:,k]
                end

                # Scale first column
                Akkinv = inv(A[k,k])
                for i = k+1:m
                    A[i,k] *= Akkinv
                end
            elseif info == 0
                info = k
            end

            # Update the rest of the matrix
            for j = k+1:n
                for i = k+1:m
                    A[i,j] -= A[i,k] * A[k,j]
                end
            end
        end
    end

    if pivot === NoPivot()
        # Use a negative value to distinguish a failed factorization (zero in pivot
        # position during unpivoted LU) from a valid but rank-deficient factorization
        info = -info
    end

    check && _check_lu_success(info, allowsingular)

    # Return LU object with row and column pivots
    return LU{T,typeof(A),typeof(ipiv)}(A, ipiv, jpiv, convert(LinearAlgebra.BlasInt, info))
end


A = rand(10, 10)
#B = copy(A);
#DLA_A = DLAMatrix{Float64}(A)
#F =LinearAlgebra.lu!(DLA_A, CompletePivoting())
#F = LinearAlgebra.generic_lufact!(DLA_A, CompletePivoting())

F= generic_lufact!(copy(A), RowMaximum())
m, n = size(F.factors)
L = tril(F.factors[1:m, 1:min(m,n)])
for i in 1:min(m,n); L[i,i] = 1 end
U = triu(F.factors[1:min(m,n), 1:n])
p = LinearAlgebra.ipiv2perm(F.ipiv,m)
q = LinearAlgebra.ipiv2perm(F.jpiv, n)
L * U ≈ A[p, q]
norm(L * U) ≈ norm(A[p, q])


using Revise
using LinearAlgebra
using DLA


A = rand(20, 20)
B = copy(A);
DLA_A = DLAMatrix{Float64}(A)
F =LinearAlgebra.lu!(DLA_A, NoPivot())
#F = LinearAlgebra.generic_lufact!(DLA_A, NoPivot())
m, n = size(F.factors)
L = tril(F.factors[1:m, 1:min(m,n)])
for i in 1:min(m,n); L[i,i] = 1 end
U = triu(F.factors[1:min(m,n), 1:n])
p = LinearAlgebra.ipiv2perm(F.ipiv,m)
q = LinearAlgebra.ipiv2perm(F.jpiv, n)
L * U ≈ B[p, q]
norm(L * U) ≈ norm(B[p, q])


using Revise
using LinearAlgebra
using DLA
A = rand(20, 20)
B = copy(A);
DLA_A = DLAMatrix{Float64}(A)
F =LinearAlgebra.lu!(DLA_A, CompletePivoting())
m, n = size(F.factors)
L = tril(F.factors[1:m, 1:min(m,n)])
for i in 1:min(m,n); L[i,i] = 1 end
U = triu(F.factors[1:min(m,n), 1:n])
p = LinearAlgebra.ipiv2perm(F.ipiv,m)
q = LinearAlgebra.ipiv2perm(F.jpiv, n)
L * U ≈ B[p, q]
norm(L * U) ≈ norm(B[p, q])
