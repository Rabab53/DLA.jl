using LinearAlgebra 
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc

function geqrt!(::Type{T}, A::AbstractMatrix{T}, Tau::AbstractMatrix{T}) where {T<: Number}
    m,n = size(A)
    nb, nt = size(Tau)
    lda = max(1,m)
    ldt = max(1,nb)

    work = Vector{T}(undef, nb*n)
    info = Ref{BlasInt}()

    if T == ComplexF64
        ccall((@blasfunc(zgeqrt_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            m, n, nb, A, lda, Tau, ldt, work, info)

    elseif T == Float64
        ccall((@blasfunc(dgeqrt_), libblastrampoline), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            m, n, nb, A, lda, Tau, ldt, work, info)

    elseif T == ComplexF32
        ccall((@blasfunc(cgeqrt_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            m, n, nb, A, lda, Tau, ldt, work, info)

    else #T  = Float32
        ccall((@blasfunc(sgeqrt_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            m, n, nb, A, lda, Tau, ldt, work, info)
    end
end

function larf!(::Type{T}, side::AbstractChar, V::AbstractVector{T}, Tau, C::AbstractMatrix{T}) where {T<: Number}
    m,n = size(C)
    incv = 1
    ldc = max(1,m)

    if side == 'L'
        work = Vector{T}(undef, n)
    else
        work = Vector{T}(undef, m)
    end

    if T == ComplexF64
        ccall((@blasfunc(zlarf_), libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ref{T}, Ptr{T}, Ref{BlasInt}, Ptr{T}),
            side, m, n, V, incv, Tau, C, ldc, work)
    elseif T == Float64
        ccall((@blasfunc(dlarf_), libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ref{T}, Ptr{T}, Ref{BlasInt}, Ptr{T}),
            side, m, n, V, incv, Tau, C, ldc, work)
    elseif T == ComplexF32
        ccall((@blasfunc(clarf_), libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ref{T}, Ptr{T}, Ref{BlasInt}, Ptr{T}),
            side, m, n, V, incv, Tau, C, ldc, work)
    else # T = Float32
        ccall((@blasfunc(slarf_), libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ref{T}, Ptr{T}, Ref{BlasInt}, Ptr{T}),
            side, m, n, V, incv, Tau, C, ldc, work)
    end
end

function larfg!(::Type{T}, alpha::T, X::AbstractVector{T}, Tau::T) where{T<: Number}
    n = length(X) + 1
    incx = 1

    println(alpha)

    if T == ComplexF64
        ccall((@blasfunc(zlarfg_), libblastrampoline), Cvoid,
        (Ref{BlasInt}, Ref{T}, Ptr{T}, Ref{BlasInt}, Ref{T}),
            n,alpha,X,incx,Tau)
    elseif T == Float64
        ccall((@blasfunc(dlarfg_), libblastrampoline), Cvoid,
        (Ref{BlasInt}, Ref{T}, Ptr{T}, Ref{BlasInt}, Ref{T}),
            n,alpha,X,incx,Tau)
    end

    println("ret from larfg ", alpha)
    return alpha, Tau
end
function geqr2!(::Type{T}, A::AbstractMatrix{T}, Tau::AbstractVector{T}) where {T<: Number}
    m,n = size(A)
    lda = max(1,m)

    work = Vector{T}(undef, n)
    info = Ref{BlasInt}()

    if T == ComplexF64
        ccall((@blasfunc(zgeqr2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{T}, Ref{BlasInt}),
            m, n, A, lda, Tau, work, info)

    elseif T == Float64
        ccall((@blasfunc(dgeqr2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{T}, Ref{BlasInt}),
            m, n, A, lda, Tau, work, info)

    elseif T == ComplexF32
        ccall((@blasfunc(cgeqr2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{T}, Ref{BlasInt}),
            m, n, A, lda, Tau, work, info)

    else #T  = Float32
        ccall((@blasfunc(sgeqr2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{T}, Ref{BlasInt}),
            m, n, A, lda, Tau, work, info)
    end
end


function larft!(::Type{T}, direct::AbstractChar, storev::AbstractChar, V::AbstractMatrix{T},
    Tau::AbstractVector{T}, Tee::AbstractMatrix{T}) where {T<: Number}

    ldt, k = size(Tee)
    ldv, dv = size(V)

    if storev == 'C'
        n = ldv
    else
        n = dv
    end

    if T == ComplexF64
        ccall((@blasfunc(zlarft_), libblastrampoline), Cvoid,
        ( Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ptr{T}, Ref{BlasInt}),
            direct, storev, n, k, V, ldv, Tau, Tee, ldt)

    elseif T == Float64
        ccall((@blasfunc(dlarft_), libblastrampoline), Cvoid,
        ( Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ptr{T}, Ref{BlasInt}),
            direct, storev, n, k, V, ldv, Tau, Tee, ldt)

    elseif T == ComplexF32
        ccall((@blasfunc(clarft_), libblastrampoline), Cvoid,
        ( Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ptr{T}, Ref{BlasInt}),
            direct, storev, n, k, V, ldv, Tau, Tee, ldt)

    else #T  = Float32
        ccall((@blasfunc(slarft_), libblastrampoline), Cvoid,
        ( Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ptr{T}, Ref{BlasInt}),
            direct, storev, n, k, V, ldv, Tau, Tee, ldt)
    end
end