using LinearAlgebra
using BenchmarkTools

include("../src/getrf.jl")

function gen_getrf2_time(::Type{T}, m, n) where {T<: Number}
    A = rand(T,m,n)
    ipiv = ones(Int, min(m,n))
    info = Ref{Int}(0)

    for i in 1:min(m,n)
        ipiv[i] = Int(i)
    end

    B = deepcopy(A)
    bpiv = deepcopy(ipiv)

    tj = @belapsed getrf2!($A, $ipiv, $info) evals = 1
    tl = @belapsed lapack_getrf2!($T, $B, $bpiv) evals = 1

    return tj, tl
end

BLAS.set_num_threads(Threads.nthreads())
xvals = Int[]
yj = Float64[]
yl = Float64[]

for m in [1000, 2197, 4096, 9214]
    push!(xvals, m)

    for T in [Float64, Float32]
        n = m
        tj, tl = gen_getrf2_time(T, m, n)

        println("m is ", m, " julia time: ", tj, " lapack time: ", tl)
        push!(yj, tj)
        push!(yl, tl)
    end
end

#display(xvals)
#display(yj)
#display(yl)
