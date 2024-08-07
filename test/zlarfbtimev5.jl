using LinearAlgebra
using BenchmarkTools
using Plots
using StatsPlots
using JLD2

include("zlarfbwrap.jl")
include("zlarfb_v0.jl")
include("zlarfb_v1.jl")
include("zlarfb_v2.jl")
include("zlarfb_v3.jl")

#BLAS.set_num_threads(1) # to make sequential
t = 40
BLAS.set_num_threads(Threads.nthreads())

for T in [Float64, ComplexF64, Float32, ComplexF32]
    println(T)

    for k in [64, 128]
        for trans in ['C', 'N']

            println("k is ", k, " trans is ", trans)

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
            
            #k = 128
            #trans = 'N'
            
            for m in [512, 1024, 2048, 4096, 8192, 16384, 32768]
                println("m = ", m)
                n = m
                
                push!(xvals, m)
                
                #T = Float64
            
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
                
                s = @belapsed larfb!($T,$side,$trans,$direct,$storev,$V,$Tau,$C) samples = 7 evals = 1
                sm = @ballocated larfb!($T,$side,$trans,$direct,$storev,$V,$Tau,$C) samples = 7 evals = 1
                push!(y, s)
                push!(ym, sm / 10^3)
                
                s0 = @belapsed zlarfbv0($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c0, $m, $work, $ldw) samples = 7 evals = 1
                s0m = @ballocated zlarfbv0($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c0, $m, $work, $ldw) samples = 7 evals = 1
                push!(y0, s0)
                push!(y0m, s0m / 10^3)
            
                s1 = @belapsed zlarfbv1($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c1, $m, $work, $ldw) samples = 7 evals = 1
                s1m = @ballocated zlarfbv1($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c1, $m, $work, $ldw) samples = 7 evals = 1
                push!(y1, s1)
                push!(y1m, s1m / 10^3)
            
                s2 = @belapsed zlarfbv2($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c2, $m, $work, $ldw) samples = 7 evals = 1
                s2m = @ballocated zlarfbv2($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c2, $m, $work, $ldw) samples = 7 evals = 1
                push!(y2, s2)
                push!(y2m, s2m / 10^3)
            
                s3 = @belapsed zlarfbv3($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c3, $m, $work, $ldw) samples = 7 evals = 1
                s3m = @ballocated zlarfbv3($side, $trans, $direct, $storev, $m, $n, $k, $V, $ldv, $Tau, $k, $c3, $m, $work, $ldw) samples = 7 evals = 1
                push!(y3, s3)
                push!(y3m, s3m / 10^3)
                
                push!(rd, s0/s1)
                push!(rf, s3/s2)
                push!(rdvl, s1/s)
                push!(rfvl, s3/s)
                push!(rlvd, s/s1)
                push!(rlvf, s/s3)
            
            end
            
            # plotting, can do later too if needed

            xvals = Int.(xvals)
            xvals = string.(xvals)

            @save "larfb type=$T t=$t k=$k trans=$trans.jdl2" xvals y ym y0 y0m y1 y1m y2 y2m y3 y3m rd rf rdvl rfvl rlvd rlvf
            
            p = plot()
            plot!(p, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel = "Time (s)", title="larfb Time Single-Thread")
            plot!(p, xvals, y, marker=:circle, label="lapack")
            plot!(p, xvals, y1, marker=:star3, label="Multiple Dispatch")
            plot!(p, xvals, y3, marker=:star5, label="Internal Function")
            savefig(p, "larfb time type=$T t=$t k=$k trans=$trans")
            
            q = plot()
            plot!(q, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel = "Memory (KB)", title="larfb Memory Single-Thread")
            # plot!(q, xvals, ym, marker=:circle, label="lapack")
            plot!(q, xvals, y1m, marker=:star3, label="Multiple Dispatch")
            plot!(q, xvals, y3m, marker=:star5, label="Internal Function")
            savefig(q, "larfb memory type=$T t=$t k=$k trans=$trans")
            
            """
            p5 = plot()
            plot!(p5, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel = "julia / lapack", title="larfb Time Ratios Single-Thread")
            plot!(p5, xvals, rdvl, marker=:circle, label="Multiple Dispatch")
            plot!(p5, xvals, rfvl, marker=:circle, label="Internal Function")
            savefig(p5, "larfb time ratios type=$T t=$t k=$k trans=$trans")
            """
            rats = [rlvd; rlvf]
            b1 = minimum(rats) - 0.15
            b2 = maximum(rats) + 0.15

            p1 = groupedbar(xvals, [rlvd rlvf], ylimits=(b1,b2),
                            label=["Multiple Dispatch" "Internal Function"], 
                            xlabel="Matrix Size (n x n)", ylabel="lapack / julia", title="time ratios")
            savefig(p1, "larfb time ratio type=$T t=$t k=$k trans=$trans")


        end
    end
end

