using Test
using LinearAlgebra

include("zlarfbwrap.jl")
include("zlarfb_v3.jl")
include("zlarfb_v1.jl")

function gen_zlarfb_test_rand(::Type{T}, side, trans, storev, direct, m, n, k) where {T<: Number}
    C = rand(T, m, n)
    D = deepcopy(C)
    Tau = rand(T, k, k)
    Tau1 = deepcopy(Tau)

   # if direct == 'F'
      #  Tau = UpperTriangular(rand(T, k, k))
   # else 
      #  Tau = LowerTriangular(rand(T, k, k))
   # end

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
        work =  rand(T, m, k)
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
                #Tau1 = UpperTriangular(Tau1)

                if storev == 'C'
                    V[j,i] = 0
                else
                    V[i,j] = 0
                end

            else
                Tau[j,i] = 0
                #Tau1 = LowerTriangular(Tau1)

                if storev == 'C'
                    V[dv - k + i, j] = 0
                else
                    V[j, dv - k + i] = 0
                end
            end
        end
    end

    if direct == 'F'
        Tau1 = UpperTriangular(Tau1)
    else
        Tau1 = LowerTriangular(Tau1)
    end

    C .*= 1000
    D .*= 1000

    larfb!(T,side,trans,direct,storev,V,Tau,C)
    zlarfbcopyv2(side, trans, direct, storev, m, n, k, V, ldv,Tau,k,D,m,work,ldw)
    
    """
    numdiffs = 0 
    maxdiff = 0+0im

    for i in 1:m
        for j in 1:n
            if C[i,j] != D[i,j]
                numdiffs = numdiffs + 1 
                diff = C[i,j] - D[i,j]
                if abs(maxdiff) > abs(diff) || maxdiff == 0
                    maxdiff = diff
                end
            end
        end
    end
    """

    #return norm((C-D)./C)
    return norm((C-D)./C)/(sqrt(2*m*n))
end

@testset "minitest" begin
    # @test gen_zlarfb_test_rand(ComplexF64, 'L' , 'N', 'C', 'F', 5,5,3) ≈ 0 atol=5e-15
    @test gen_zlarfb_test_rand(ComplexF64, 'L' , 'N', 'C', 'F', 10,12,8) ≈ 0 atol=5e-15
    @test gen_zlarfb_test_rand(ComplexF64, 'L' , 'N', 'C', 'F', 100,115,85) ≈ 0 atol=5e-15
end

@testset "datatype = Float64" begin
    for storev in ['C', 'R']
        for direct in ['F', 'B']
            for side in ['L', 'R']
                @testset "storev=$storev, direct=$direct, side=$side, trans=$trans" for trans in ['N', 'C']
                    for m in [50, 100, 200, 500,1000]
                        for n in [m, div(m,10)*9, div(m,10)*11]
                           # @testset "m=m, n=n, k=k" 
                           for k in [min(m,n), div(min(m,n), 5)*4, div(min(m,n), 5)*3]
                                @test gen_zlarfb_test_rand(Float64,side, trans, storev, direct, m,n,k) ≈ 0 atol=1e-15
                                #@test gen_zlarfb_test_rand(ComplexF64,side, trans, storev, direct, m,n,k) ≈ 0 atol=1e-15
                                #@test gen_zlarfb_test_rand(Float32,side, trans, storev, direct, m,n,k) ≈ 0 atol=1e-7
                                #@test gen_zlarfb_test_rand(ComplexF32,side, trans, storev, direct, m,n,k) ≈ 0 atol=1e-7
                            end
                        end
                    end
                end
            end
        end
    end
end

"""
@testset "zlarfb_test_rand" begin
    
    @testset "datatype = T" for T in [ComplexF64, Float64]
        for storev in ['C', 'R']
            for direct in ['F', 'B']
                for side in ['L', 'R']
                    @testset "storev=storev, direct=direct, side=side, trans=trans" for trans in ['N', 'C'] 
                         for m in [10, 50, 100, 200, 500]
                             for n in [m, div(m,10)*9, div(m,10)*11]
                                for k in [min(m,n), div(min(m,n), 5)*4, div(min(m,n), 5)*3]
                                    @test gen_zlarfb_test_rand(T, side, trans, storev, direct, m,n,k) ≈ 0 atol=1e-13
                                    #@test gen_zlarfb_test_rand(T, side, trans, storev, direct, m,n,k)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    #@testset "datatype = T" for T in [ComplexF32, Float32]
        #for storev in ['C', 'R']
            #for direct in ['F', 'B']
                #for side in ['L', 'R']
                    # @testset "storev=storev, direct=direct, side=side, trans=trans" for trans in ['N', 'C'] 
                        #for m in [10, 50, 100, 200, 500]
                            #for n in [m, div(m,10)*9, div(m,10)*11]
                                #for k in [min(m,n), div(min(m,n), 5)*4, div(min(m,n), 5)*3]
                                    #@test gen_zlarfb_test_rand(T, side, trans, storev, direct, m,n,k) ≈ 0 atol=2e-5
                                    #@test gen_zlarfb_test_rand(T, side, trans, storev, direct, m,n,k) 
                                #end
                            #end
                        #end
                    #end
                #end
            #end
        #end
    #end
end
"""