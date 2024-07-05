using LinearAlgebra
using BenchmarkTools
using Plots

include("zlarfbwrap.jl")
include("zlarfb_v0.jl")
include("zlarfb_v1.jl")
include("zlarfb_v2.jl")
include("zlarfb_v3.jl")

BLAS.set_num_threads(1) # to make sequential

xvals = Float64[]
y = Float64[]
ym = Float64[]

y0 = Float64[]
y0m = Float64[]

y1 = Float64[]
y1m = Float64[]

y2 = Float64[]
y2m = Float64[]

y3 = Float64[]
y3m = Float64[]


for m in [512, 1024, 2048, 4096, 8192, 16384]
    n = m
    k = 64

    push!(xvals, m)
    
    T = Float64

    storev = 'C'
    direct = 'F'
    side = 'L'
    trans = 'C'
                    
    C = rand(T, m, n)
    Tau = rand(T, k, k)

    if side == 'L'
        work = rand(T,n,k)
        ldw = n

        if storev == 'C'
            V = rand(T,m,k)
            ldv = m
            dv = m
        else #storev = R
            V = rand(T,k,m)
            ldv = k
            dv = m
        end

    else #side = 'R'
        work = rand(T, m, k)
        ldw = m

        if storev == 'C'
            V = rand(T,n,k)
            ldv = n
            dv = n
        else #storev = R
            V = rand(T,k,n)
            ldv = k
            dv = n
        end
    end

    for i in 1:k
        if direct == 'F'
            V[i,i] = 1
        else
            if storev == 'C'
                V[dv - k + i, i] = 1
            else
                V[i, dv - k + i] = 1
            end
        end

        for j in 1:(i-1)
            if direct == 'F' #Tau is upper triangular 
                Tau[i,j] = 0

                if storev == 'C'
                    V[j,i] = 0
                else
                    V[i,j] = 0
                end

            else
                Tau[j,i] = 0

                if storev == 'C'
                    V[dv - k + i, j] = 0
                else
                    V[j, dv - k + i] = 0
                end
            end
        end
    end

    c0 = deepcopy(C)
    c1 = deepcopy(C)
    c2 = deepcopy(C)
    c3 = deepcopy(C)
    
    s = @belapsed larfb!(Float64,$side,$trans,$direct,$storev,$V,$Tau,$C) samples = 5 evals = 1
    sm = @ballocated larfb!(Float64,$side,$trans,$direct,$storev,$V,$Tau,$C) samples = 5 evals = 1
    push!(y, s)
    push!(ym, sm / 10^3)
    
    s0 = @belapsed zlarfbv0($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c0, $m, $work, $ldw) samples = 5 evals = 1
    s0m = @ballocated zlarfbv0($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c0, $m, $work, $ldw) samples = 5 evals = 1
    push!(y0, s0)
    push!(y0m, s0m / 10^3)

    s1 = @belapsed zlarfbv1($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c1, $m, $work, $ldw) samples = 5 evals = 1
    s1m = @ballocated zlarfbv1($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c1, $m, $work, $ldw) samples = 5 evals = 1
    push!(y1, s1)
    push!(y1m, s1m / 10^3)

    s2 = @belapsed zlarfbv2($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c2, $m, $work, $ldw) samples = 5 evals = 1
    s2m = @ballocated zlarfbv2($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c2, $m, $work, $ldw) samples = 5 evals = 1
    push!(y2, s2)
    push!(y2m, s2m / 10^3)

    s3 = @belapsed zlarfbv3($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c3, $m, $work, $ldw) samples = 5 evals = 1
    s3m = @ballocated zlarfbv3($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c3, $m, $work, $ldw) samples = 5 evals = 1
    push!(y3, s3)
    push!(y3m, s3m / 10^3)

end

p = plot()
plot!(p, legend=:topleft, xlabel="matrix size", ylabel = "time (s)", title="larfb time single thread")
plot!(p, xvals, y, marker=:circle, label="lapack")
plot!(p, xvals, y0, marker=:circle, label="multiple dispatch + no wrap")
plot!(p, xvals, y1, marker=:circle, label="multiple dispatch + wrap")
plot!(p, xvals, y2, marker=:circle, label="internal function  + wrap")
plot!(p, xvals, y3, marker=:circle, label="internal function + no wrap")
savefig(p, "test time plot")

q = plot()
plot!(q, legend=:topleft, xlabel="matrix size", ylabel = "memory (KB))", title="larfb memory single thread")
plot!(q, xvals, ym, marker=:circle, label="lapack")
plot!(q, xvals, y0m, marker=:circle, label="multiple dispatch + no wrap")
plot!(q, xvals, y1m, marker=:circle, label="multiple dispatch + wrap")
plot!(q, xvals, y2m, marker=:circle, label="internal function  + wrap")
plot!(q, xvals, y3m, marker=:circle, label="internal function + no wrap")
savefig(q, "test memory plot")