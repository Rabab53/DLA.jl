using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools

function axpy!(a, x, y)
    n = length(x)

    if n <= 0
        return
    end

    if abs(real(a)) + abs(imag(a)) == zero(eltype(x))
        return 
    end

    for i in 1:n
        y[i] = y[i] + a*x[i]
    end

    return
end