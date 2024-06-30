using DLA
using Test
using Random

# @testset "DLA.jl" begin
#     # Write your tests here.
# end

function test_approx_equal(A, B, err)
    @assert size(A) == size(B)
    d = abs.((A .- B) ./ minimum(size(A)))
    ok = all(x->x<=err, d)
    # if !ok 
    #     display(A)
    #     display(B)
    #     display(d)
    # end
    return ok
end

for (elty, err) in
    ((Float64,   1e-16),
     (Float32,   1e-8),
     (ComplexF64,1e-16*sqrt(2)),
     (ComplexF32,1e-8*sqrt(2)))
    @eval begin
        function test_approx_equal(
            A::Union{AbstractMatrix{$elty}, Vector{$elty}}, 
            B::Union{AbstractMatrix{$elty}, Vector{$elty}})
            return test_approx_equal(A, B, $err)
        end
    end
end

@testset "core_zgbtype1cb" begin
    @testset for elty in (Float32, Float64, ComplexF32, ComplexF64)
        uplo = DLA.CoreBlasUpper
        n=8
        nb=4
        # figure out acceptable st, ed given n, nb
        st=1
        ed=4
        sweep=1
        Vblksiz=1
        wantz=0

        Random.seed!(0)
        A1 = rand(3*nb+1, n)
        VP1 = zeros(n)
        VQ1 = zeros(n)
        TAUP1 = zeros(n)
        TAUQ1 = zeros(n)
        DLA.ref_coreblas_gbtype1cb!(uplo, n, nb, A1, VQ1, TAUQ1, VP1, TAUP1, st, ed, sweep, Vblksiz, wantz)

        Random.seed!(0)
        A2 = rand(Float64, (3*nb+1, n))
        VP2 = zeros(Float64, n)
        VQ2 = zeros(Float64, n)
        TAUP2 = zeros(Float64, n)
        TAUQ2 = zeros(Float64, n)
        DLA.coreblas_gbtype1cb!(uplo, n, nb, A2, VQ2, TAUQ2, VP2, TAUP2, st, ed, sweep, Vblksiz, wantz)

        @test test_approx_equal(A1, A2)
        @test test_approx_equal(VP1, VP2)
        @test test_approx_equal(VQ1, VQ2)
        @test test_approx_equal(TAUP1, TAUP2)
        @test test_approx_equal(TAUQ1, TAUQ2)

    end
end

