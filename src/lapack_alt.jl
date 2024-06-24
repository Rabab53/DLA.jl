# extended from https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/lapack.jl

using LinearAlgebra.BLAS: @blasfunc, chkuplo
using LinearAlgebra.LAPACK: chkside

using LinearAlgebra: libblastrampoline, BlasFloat, BlasInt, LAPACKException, DimensionMismatch,
    SingularException, PosDefException, chkstride1, checksquare, triu, tril, dot

## alternative arguments for coreblas compatibility?

## Tools to compute and apply elementary reflectors
for (larfg, elty) in
    ((:dlarfg_, Float64),
     (:slarfg_, Float32),
     (:zlarfg_, ComplexF64),
     (:clarfg_, ComplexF32))
    @eval begin
        #        .. Scalar Arguments ..
        #        INTEGER            incx, n
        #        DOUBLE PRECISION   alpha, tau
        #        ..
        #        .. Array Arguments ..
        #        DOUBLE PRECISION   x( * )
        function LinearAlgebra.LAPACK.larfg!(n::Integer, α::Ref{$elty}, x::Ptr{$elty}, τ::Ref{$elty})
            require_one_based_indexing(x)
            incx = BlasInt(1)
            ccall((@blasfunc($larfg), libblastrampoline), Cvoid,
                (Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty}),
                BlasInt(n), α, x, incx, τ)
            # return τ[]
        end
    end
end

for (larf, elty) in
    ((:dlarf_, Float64),
     (:slarf_, Float32),
     (:zlarf_, ComplexF64),
     (:clarf_, ComplexF32))
    @eval begin
        #        .. Scalar Arguments ..
        #        CHARACTER          side
        #        INTEGER            incv, ldc, m, n
        #        DOUBLE PRECISION   tau
        #        ..
        #        .. Array Arguments ..
        #        DOUBLE PRECISION   c( ldc, * ), v( * ), work( * )
        function LinearAlgebra.LAPACK.larf!(side::AbstractChar, m::Integer, n::Integer, v::Ptr{$elty},
                       τ::$elty, C::Ptr{$elty}, ldc::Integer, work::AbstractVector{$elty})
            require_one_based_indexing(v, C, work)
            chkside(side)
            incv  = BlasInt(1)
            ccall((@blasfunc($larf), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Clong),
                side, m, n, v, incv,
                τ, C, ldc, work, 1)
            # return C
        end
    end
end