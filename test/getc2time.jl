using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools

include("../src/getc2.jl")

function gen_getc2_time(::Type{T}, n) where T
    A = rand(T,n,n)

    ipiv = ones(Int, n)
    jpiv = ones(Int, n)
    info = Ref{Int}(0)

    for i in 1:n
        ipiv[i] = Int(i)
        jpiv[i] = Int(i)
    end

    B = deepcopy(A)
    bipiv = deepcopy(ipiv)
    bjpiv = deepcopy(jpiv)

    
    C = deepcopy(A)
    cipiv = deepcopy(ipiv)
    cjpiv = deepcopy(jpiv)
    cinfo = Ref{Int}(0)

    tj = @belapsed getc2!($A,$ipiv,$jpiv,$info) evals = 1
    tl = @belapsed lapack_getc2!($B,$bipiv,$bjpiv) evals = 1
    mj = @ballocated getc2!($C,$cipiv,$cjpiv,$cinfo) evals = 1

    return tj, tl, mj
end


BLAS.set_num_threads(Threads.nthreads())
println("threads should be ", Threads.nthreads())
xvals = Int[]
yj = Float64[]
yl = Float64[]

for n in [250, 500, 1000, 1500]
    push!(xvals, n)

    for T in [Float64, Float32]

        tj, tl, mj = gen_getc2_time(T, n)
        println("n is ", n, " julia time: ", tj, " lapack time: ", tl, " julia memory: ", mj)
        push!(yj, tj)
        push!(yl, tl)
    end
end
