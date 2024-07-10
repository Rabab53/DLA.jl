using LinearAlgebra
using BenchmarkTools
using Plots
using JLD2

include("zlarfbwrap.jl")
include("zlarfb_v0.jl")
include("zlarfb_v1.jl")
include("zlarfb_v2.jl")
include("zlarfb_v3.jl")

#BLAS.set_num_threads(1) # to make sequential
BLAS.set_num_threads(Threads.nthreads())

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

rd = Float64[]
rf = Float64[]

rdvl = Float64[]
rfvl = Float64[]
rlvd = Float64[]
rlvf = Float64[]

k = 128
trans = 'N'

for m in [512, 1024, 2048, 4096, 8192, 16384, 32768]
    println("m = ", m)
    n = m
    
    push!(xvals, m)
    
    T = Float64

    storev = 'C'
    direct = 'F'
    side = 'L'
                    
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
    
    push!(rd, s0/s1)
    push!(rf, s3/s2)
    push!(rdvl, s1/s)
    push!(rfvl, s3/s)
    push!(rlvd, s/s1)
    push!(rlvf, s/s3)

end

@save "larfb t=40 k=$k trans=$trans.jdl2" xvals y ym y0 y0m y1 y1m y2 y2m y3 y3m rd rf rdvl rfvl rlvd rlvf

p = plot()
plot!(p, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel = "Time (s)", title="larfb Time Multi-Thread")
plot!(p, xvals, y, marker=:circle, label="lapack")
plot!(p, xvals, y1, marker=:circle, label="Multiple Dispatch")
plot!(p, xvals, y3, marker=:circle, label="Internal Function")
savefig(p, "larfb time t=40 trans=$trans k=$k")

q = plot()
plot!(q, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel = "Memory (KB)", title="larfb Memory Multi-Thread")
# plot!(q, xvals, ym, marker=:circle, label="lapack")
plot!(q, xvals, y1m, marker=:circle, label="Multiple Dispatch")
plot!(q, xvals, y3m, marker=:circle, label="Internal Function")
savefig(q, "larfb memory t=40 trans=$trans k=$k")

#p1 = plot()
#plot!(p1, legend=:topleft, xlabel="matrix size", ylabel = "time (s)", title="larfb time lapack vs internal function")
#plot!(p1, xvals, y, marker=:circle, label="lapack")
#plot!(p1, xvals, y3, marker=:circle, label="internal function")
#savefig(p1, "lapack vs function trans =$trans k=$k")

#p2 = plot()
#plot!(p2, legend=:topleft, xlabel="matrix size", ylabel = "time (s)", title="larfb time lapack vs multiple dispatch")
#plot!(p2, xvals, y, marker=:circle, label="lapack")
#plot!(p2, xvals, y1, marker=:circle, label="multiple dispatch")
#savefig(p2, "lapack vs multiple dispatch wrap k=64 trans = C")

#p4 = plot()
#plot!(p4, legend=:topleft, xlabel="matrix size", ylabel = "unwrapped / wrapped", title="larfb time wrapping comparisons")
#plot!(p4, xvals, rd, marker=:circle, label="multiple dispatch")
#plot!(p4, xvals, rf, marker=:circle, label="internal function")
#savefig(p4, "wrapping comparisons ratio k=128 trans=C")

p5 = plot()
plot!(p5, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel = "julia / lapack", title="larfb Time Ratios Multi-Thread")
plot!(p5, xvals, rdvl, marker=:circle, label="Multiple Dispatch")
plot!(p5, xvals, rfvl, marker=:circle, label="Internal Function")
savefig(p5, "time ratios t=40 k=$k trans=$trans")

#p6 = plot()
#plot!(p6, legend=:topleft, xlabel="matrix size", ylabel = "julia / lapack", title="larfb time time ratio")
#plot!(p6, xvals, rfvl, marker=:circle, label="internal function")
#savefig(p6, "time ratio internal k=64 trans=C")

