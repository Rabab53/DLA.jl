using DLA
using Test
using Random
using LinearAlgebra

# @testset "DLA.jl" begin
#     # Write your tests here.
# end

function test_approx_equal(A, B, err, verbose=false)
    @assert size(A) == size(B)
    d = abs.((A .- B) / minimum(size(A)))
    ok = all(x->x<=err, d)
    if verbose
        if !ok 
            display(A)
            display(B)
            display(d)
        end
        display("max normalized error: $(maximum(d))")
        display("    acceptable error: $(err)")
    end
    return ok
end

for (elty, err) in
    ((Float64,   1e-16),
     (Float32,   1e-8),
     (ComplexF64,1e-16*sqrt(2)),
     (ComplexF32,1e-8*sqrt(2)))
    @eval begin
        function test_approx_equal(
            A::VecOrMat{$elty}, 
            B::VecOrMat{$elty})
            return test_approx_equal(A, B, $err, true)
        end
    end
end



@testset "larfg" begin
    @testset for elty in (Float32, Float64, ComplexF32, ComplexF64)
    # @testset for elty in (ComplexF32, ComplexF64)
        ## larfg
        Random.seed!(0)
        alpha = Ref{elty}(rand(elty))
        tau = Ref{elty}(0)
        x  = rand(elty, 5)

        beta = Ref{elty}(alpha[])
        v  = copy(x)
        DLA.larfg!(beta, v, tau)
        beta1, v1, tau1 = beta[], copy(v), tau[]
        # LHS = convert(Vector{elty}, H' * [alpha[]; x])
        # RHS = convert(Vector{elty}, [beta[]; zeros(5)])
        # @test test_approx_equal(LHS, RHS)

        # todo: lapack.jl has some tests

        beta = Ref{elty}(alpha[])
        v  = copy(x)
        LAPACK.larfg!(6, beta, pointer(v,1), tau)
        beta2, v2, tau2 = beta[], copy(v), tau[]
        # H = I - tau[] * [1; v] * [1; v]'
        # LHS = convert(Vector{elty}, H' * [alpha[]; x])
        # RHS = convert(Vector{elty}, [beta[]; zeros(5)])
        # @test test_approx_equal(LHS, RHS)

        @test isapprox(beta1, beta2)
        @test test_approx_equal(v1, v2)
        @test isapprox(tau1, tau2)
    end
end

# @testset "coreblas_gbtype1cb" begin
#     @testset for elty in (Float32, Float64, ComplexF32, ComplexF64)
#         uplo = DLA.CoreBlasLower
#         n=12
#         nb=4
#         # figure out acceptable st, ed given n, nb
#         st=1
#         ed=5
#         sweep=1
#         Vblksiz=1
#         wantz=0

#         Random.seed!(0)
#         A1 = rand(elty, (3*nb+1, n))
#         VP1 = zeros(elty, n)
#         VQ1 = zeros(elty, n)
#         TAUP1 = zeros(elty, n)
#         TAUQ1 = zeros(elty, n)
#         DLA.ref_coreblas_gbtype1cb!(uplo, n, nb, A1, VQ1, TAUQ1, VP1, TAUP1, st, ed, sweep, Vblksiz, wantz)

#         Random.seed!(0)
#         A2 = rand(elty, (3*nb+1, n))
#         VP2 = zeros(elty, n)
#         VQ2 = zeros(elty, n)
#         TAUP2 = zeros(elty, n)
#         TAUQ2 = zeros(elty, n)
#         DLA.coreblas_gbtype1cb!(uplo, n, nb, A2, VQ2, TAUQ2, VP2, TAUP2, st, ed, sweep, Vblksiz, wantz)

#         @test test_approx_equal(A1, A2)
#         @test test_approx_equal(VP1, VP2)
#         @test test_approx_equal(VQ1, VQ2)
#         @test test_approx_equal(TAUP1, TAUP2)
#         @test test_approx_equal(TAUQ1, TAUQ2)

#     end
# end


# @testset "coreblas_gbtype3cb" begin
#     @testset for elty in (Float32, Float64, ComplexF32, ComplexF64)
#         uplo = DLA.CoreBlasLower
#         n=12
#         nb=4
#         # figure out acceptable st, ed given n, nb
#         st=1
#         ed=5
#         sweep=1
#         Vblksiz=1
#         wantz=0

#         Random.seed!(0)
#         A1 = rand(elty, (3*nb+1, n))
#         VP1 = zeros(elty, n)
#         VQ1 = zeros(elty, n)
#         TAUP1 = zeros(elty, n)
#         TAUQ1 = zeros(elty, n)
#         DLA.ref_coreblas_gbtype3cb!(uplo, n, nb, A1, VQ1, TAUQ1, VP1, TAUP1, st, ed, sweep, Vblksiz, wantz)

#         Random.seed!(0)
#         A2 = rand(elty, (3*nb+1, n))
#         VP2 = zeros(elty, n)
#         VQ2 = zeros(elty, n)
#         TAUP2 = zeros(elty, n)
#         TAUQ2 = zeros(elty, n)
#         DLA.coreblas_gbtype3cb!(uplo, n, nb, A2, VQ2, TAUQ2, VP2, TAUP2, st, ed, sweep, Vblksiz, wantz)

#         @test test_approx_equal(A1, A2)
#         @test test_approx_equal(VP1, VP2)
#         @test test_approx_equal(VQ1, VQ2)
#         @test test_approx_equal(TAUP1, TAUP2)
#         @test test_approx_equal(TAUQ1, TAUQ2)

#     end
# end