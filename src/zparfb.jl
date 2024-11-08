using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools

include("zpamm.jl")

function lapack_tprfb!(::Type{T}, side::AbstractChar, trans::AbstractChar, direct::AbstractChar, storev::AbstractChar,
    l::Int64, V::AbstractMatrix{T}, Tee::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<: Number}

    m,n = size(B)
    ldt, k = size(Tee)
    ldv = max(1, stride(V,2))
    lda = max(1, stride(A,2))
    ldb = max(1,m)

    if side == 'L'
        ldw = k
        work = Vector{T}(undef, ldw*n)
    else
        ldw = m
        work = Vector{T}(undef, ldw*k)
    end

    if m > 0 && n > 0
        if T == ComplexF64
            ccall((@blasfunc(ztprfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            side, trans, direct, storev, m, n, k, l, V, ldv, Tee, ldt, A, lda, B, ldb, work, ldw)

        elseif T == ComplexF32
            ccall((@blasfunc(ctprfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            side, trans, direct, storev, m, n, k, l, V, ldv, Tee, ldt, A, lda, B, ldb, work, ldw)

        elseif T == Float64
            ccall((@blasfunc(dtprfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
                side, trans, direct, storev, m, n, k, l, V, ldv, Tee, ldt, A, lda, B, ldb, work, ldw)
        else # T == Float32
            ccall((@blasfunc(stprfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            side, trans, direct, storev, m, n, k, l, V, ldv, Tee, ldt, A, lda, B, ldb, work, ldw)
        end
    end
end

function zparfb(side, trans, direct, storev, m1, n1, m2, n2, k, l, 
                A1, lda1, A2, lda2, V,  ldv, T, ldt, work, ldwork)

    if side != 'L' && side != 'R'
        throw(ArgumentError("illegal value of side"))
        return -1
    end

    if trans != 'N' && trans != 'C' && trans != 'T'
        throw(ArgumentError("illegal value of trans"))
        return -2
    end

    if direct != 'F' && direct != 'B'
        throw(ArgumentError("illegal value of direct"))
        return -3
    end

    if storev != 'C' && storev != 'R'
        throw(ArgumentError("illegal value of storev"))
        return -4
    end

    if m1 < 0
        throw(ArgumentError("illegal value of m1"))
        return -5
    end

    if n1 < 0
        throw(ArgumentError("illegal value of n1"))
        return -6
    end

    if m2 < 0 || (side == 'R' && m1 != m2)
        throw(ArgumentError("illegal value of m2"))
        return -7
    end

    if n2 < 0 || (side == 'L' && n1 != n2)
        throw(ArgumentError("illegal value of n2"))
        return -8
    end

    if k < 0
        throw(ArgumentError("illegal value of k"))
        return -9
    end

    if l < 0
        throw(ArgumentError("illegal value of l"))
        return -10
    end

    if lda1 < 0
        throw(ArgumentError("illegal value of lda1"))
        return -12
    end

    if lda2 < 0
        throw(ArgumentError("illegal value of lda2"))
        return -14
    end

    if ldv < 0
        throw(ArgumentError("illegal value of ldv"))
        return -16
    end

    if ldt < 0
        throw(ArgumentError("illegal value of ldt"))
        return -18
    end

    if ldwork < 0
        throw(ArgumentError("illegal value of ldwork"))
        return -20
    end

    # quick return 

    if m1 == 0 || n1 == 0 || n2 == 0 || k == 0
        return 
    end

    one0 = oneunit(eltype(A1))
    zero0 = zero(eltype(A1))

    if trans == 'N'
        tfun = identity
    else
        tfun = adjoint
    end


    if direct == 'F'
        if side == 'L'
            # Form H * A or H^H * A 
            # w =  A1 + op(V) * A2
            zpamm('W', 'L', storev, k, n1, m2, l, A1, lda1, A2, lda2, V, ldv, work, k)

            LinearAlgebra.generic_trimatmul!((@view work[1:k, 1:n2]), 'U', 'N', tfun, (@view T[1:k, 1:k]), (@view work[1:k, 1:n2]))

            #A1 = A1 - w
            for j in 1:n1
                LinearAlgebra.axpy!(-one0, (@view work[1:k, j]), (@view A1[1:k, j]))
            end

            #a2 = A2 - op(V) * w
            zpamm('A', 'L', storev, m2, n2, k, l, A1, lda1, A2, lda2, V, ldv, work, ldwork)

        else #side = 'R'

            # W = A1 + A2 * op(V)
            zpamm('W', 'R', storev, m1, k, n2, l, A1, lda1, A2, lda2, V, ldv, work, ldwork)

            #w = w * op(T)
            
            LinearAlgebra.generic_mattrimul!((@view work[1:m2, 1:k]), 'U', 'N', tfun, (@view work[1:m2, 1:k]), (@view T[1:k, 1:k]))

            #A1 = A1 - w
            for j in 1:k
                LinearAlgebra.axpy!(-one0, (@view work[1:m1, j]), (@view A1[1:m1, j]))
            end

            #a2 = a2 - w * op(V)
            zpamm('A', 'R', storev, m2, n2, k, l, A1, lda1, A2, lda2, V, ldv, work, ldwork)
        end
    else
       throw(ErrorException("not yet supported"))
       return
    end

    return
end