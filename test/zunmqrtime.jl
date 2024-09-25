using LinearAlgebra
using BenchmarkTools
using JLD2
using Plots
using StatsPlots
using AMDGPU

include("../src/zunmqr_v0.jl")
include("../src/zunmqr_v0_amd.jl")
include("../src/zunmqrwrap.jl")

#BLAS.set_num_threads(1) # to make sequential
BLAS.set_num_threads(Threads.nthreads())
t = 40

for T in [Float32]#[Float64, Float32] #[Float16]#[Float64, Float32, ComplexF64, ComplexF32]#[Float64]#[ComplexF32] #[Float64, Float32, ComplexF64, ComplexF32]
    println(T)
    
    side = 'L'

    for ib in [128]
        for trans in ['N']#['N', 'C']
            println("ib is ", ib, " trans is ", trans)

            xvals = Float64[]

            y = Float64[] #lapack
            ym = Float64[]

            y1 = Float64[] #lapack
            ym1 = Float64[]

            y0j = Float64[] #julia native
            y0jm = Float64[]


            y1j = Float64[] #julia native
            y1jm = Float64[]

            rlvj = Float64[]
            rjvl = Float64[]

            rlvj1 = Float64[]
            rjvl1 = Float64[]
            
	    for m in [512, 1024, 2048, 4096]#, 8192, 16384]
                println("m = ", m)
                n = m
                k = m

                push!(xvals, m)

                C = rand(T,m,n)

                if side == 'L'
                    A = rand(T,m,k)
                    lda = m
                    ldw = n
                    work = rand(T,n,n)
                else
                    A = rand(T,n,k)
                    lda = n
                    ldw = m
                    work = rand(T,m,ib)
                end
                
                Tau = rand(T,ib,k)
                Tau1 = rand(T,k)
                A1 = Array(A)
                #AMDGPU.rocSOLVER.geqrf!(A, Tau)
                #AMDGPU.rocSOLVER.geqrf!(A1, Tau1)
                
                C0 = Array(C)
                C2 = Array(C)
                A2 = deepcopy(A)
                Tau2 = deepcopy(Tau)

                A3 = deepcopy(A1)
                
                C4 = deepcopy(C)

               if (T == Float64 || T == Float32) && trans == 'C'
                    #LAPACK
                    #s = @belapsed  unmqr!($T, $side, 'T', $A1, $Tau1, $C0) samples = 7 evals = 1
                    #sm = @ballocated  unmqr!($T, $side, 'T', $A1, $Tau1, $C0) samples = 7 evals = 1
                    #@show s, sm
                    #push!(y,s)
                    #push!(ym, sm / 10^3)

                    A1 = ROCArray(A1)
                    Tau1 = ROCArray(Tau1)
                    C0 = ROCArray(C0)

                    #CU
		    s1 =0.0
		    AMDGPU.rocSOLVER.ormqr!(side, 'T', A1, Tau1, C0) #samples = 7 evals = 1
		    for i in 1:50
                    s1 += @elapsed  @sync begin
		       AMDGPU.rocSOLVER.ormqr!(side, 'T', A1, Tau1, C0) #samples = 7 evals = 1
	               end
                    end
		    s1 = s1/50
		    sm1 = @ballocated  AMDGPU.rocSOLVER.ormqr!($side, 'T', $A1, $Tau1, $C0) #samples = 7 evals = 1
                    @show s1, sm1
                    push!(y1,s1)
                    push!(ym1, sm1 / 10^3)
                    AMDGPU.unsafe_free!(A1)
                    AMDGPU.unsafe_free!(Tau1)
                    AMDGPU.unsafe_free!(C0)
                else
                    #LAPACK
                    #s = @belapsed unmqr!($T, $side, $trans, $A1, $Tau1, $C0) samples = 7 evals = 1
                    #sm = @ballocated unmqr!($T, $side, $trans, $A3, $Tau1, $C0) samples = 7 evals = 1
                    #@show s, sm
                    #push!(y,s)
                    #push!(ym, sm / 10^3)

                    
                    A1 = ROCArray(A1)
                    Tau1 = ROCArray(Tau1)
                    C0 = ROCArray(C0)
                    #CU
		    s1 = 0.0
		    AMDGPU.rocSOLVER.ormqr!(side, trans, A1, Tau1, C0)
		    for i in 1:50
                    s1 += @elapsed  @sync begin
                        AMDGPU.rocSOLVER.ormqr!(side, trans, A1, Tau1, C0)
                    end
	            end
		    s1 = s1/50

                    sm1 = @ballocated  @sync begin
                        AMDGPU.rocSOLVER.ormqr!($side, $trans, $A1, $Tau1, $C0) 
                    end
                    @show s1, sm1
                    push!(y1,s1)
                    push!(ym1, sm1 / 10^3)
                    AMDGPU.unsafe_free!(A1)
                    AMDGPU.unsafe_free!(Tau1)
                    AMDGPU.unsafe_free!(C0)
                end
              

                #Julia multithreaded
                #s0j = @belapsed zunmqrv0($side, $trans, $m, $n, $k, $ib, $A, $lda, $Tau, $ib, $C, $m, $work, $ldw) samples = 7 evals = 1
                #s0jm = @ballocated zunmqrv0($side, $trans, $m, $n, $k, $ib, $A, $lda, $Tau, $ib, $C, $m, $work, $ldw) samples = 7 evals = 1
                #@show s0j, s0jm
                #push!(y0j, s0j)
                #push!(y0jm, s0jm/10^3)
  
                
                A = ROCArray(A)
                Tau = ROCArray(Tau)
                C = ROCArray(C)
                work = ROCArray(work)
                #Julia cuda
		s1j=0.0
		zunmqrv0g(side, trans, m, n, k, ib, A, lda, Tau, ib, C, m, work, ldw)
		for i in 1:50
                s1j +=  @elapsed @sync begin
                    zunmqrv0g(side, trans, m, n, k, ib, A, lda, Tau, ib, C, m, work, ldw) 
                end
	        end
                s1j = s1j/50
                s1jm = @ballocated @sync begin
                    zunmqrv0g($side, $trans, $m, $n, $k, $ib, $A, $lda, $Tau, $ib, $C, $m, $work, $ldw) 
                end
                @show s1j, s1jm
                push!(y1j, s1j)
                push!(y1jm, s1jm/10^3)

                AMDGPU.unsafe_free!(A)
                AMDGPU.unsafe_free!(Tau)
                AMDGPU.unsafe_free!(C)

                #push!(rlvj, s/s0j)
                #push!(rjvl, s0j/s)

                push!(rlvj1, s1/s1j)
                push!(rjvl1, s1j/s1)
                #AMDGPU.reclaim()
		GC.gc()

            end

            xvals = Int.(xvals)
            xvals = string.(xvals)

#            @save "unmqr type=$T t=$t ib=$ib trans=$trans.jdl2" xvals y ym  y1 ym1 y0j y0jm y1j y1jm rlvj rjvl rlvj1 rjvl1
            @save "unmqr type=$T t=$t ib=$ib trans=$trans.jdl2" xvals  ym  y1 ym1 y1j y1jm rlvj1 rjvl1

            p = plot(yaxis=:log)
            plot!(p, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel="Time (s)")
            #plot!(p, xvals, y, marker=:circle, label="LAPACK")
            #plot!(p, xvals, y0j, marker=:star8, label="Julia Multithreaded")
            plot!(p, xvals, y1, marker=:circle, label="AMDGPU.rocSOLVER")
            plot!(p, xvals, y1j, marker=:star8, label="Julia AMD-GPU")
            savefig(p, "unmqr time type=$T t=$t ib=$ib trans=$trans")

            q = plot(yaxis=:log)
            plot!(q, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel = "Memory (KB)")
            #plot!(q, xvals, ym, marker=:circle, label="LAPACK")
            #plot!(q, xvals, y0jm, marker=:star8, label="Julia Multithreaded")
            plot!(q, xvals, ym1, marker=:circle, label="AMDGPU.rocSOLVER")
            plot!(q, xvals, y1jm, marker=:star8, label="Julia AMD-GPU")
            savefig(q, "unmqr memory type=$T t=$t ib=$ib trans=$trans")
            
            rats = [rlvj; rlvj1]
            b1 = minimum(rats) - 0.15
            b2 = maximum(rats) + 0.15

            #p1 = groupedbar(xvals, [rlvj rlvj1], ylimits=(b1,b2),
            #    label=["LAPACK / Julia Multithreaded" "AMDGPU.rocSOLVER / Julia  AMD-GPU"],
            #    xlabel="Matrix Size (n x n)", ylabel="ratio", title="unmqr time ratio")
            #savefig(p1, "unmqr time ratio type=$T t=$t ib=$ib trans=$trans")


        end
    end
end
     
