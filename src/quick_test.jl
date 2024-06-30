using Random

function test_approx_equal(A, B, err)
    @assert size(A) == size(B)
    d = abs.((A .- B) ./ minimum(size(A)))
    ok = all(x->x<=err, d)
    if !ok 
        display(A)
        display(B)
        display(d)
    end
    return ok
end

for (elty, err) in
    ((:Float64,   :1e-16),
     (:Float32,   :1e-8),
     (:ComplexF64,:1e-16*sqrt(2)),
     (:ComplexF32,:1e-8*sqrt(2)))
    @eval begin
        function test_approx_equal(
            A::Union{AbstractMatrix{$elty}, Vector{$elty}}, 
            B::Union{AbstractMatrix{$elty}, Vector{$elty}})
            return test_approx_equal(A, B, $err)
        end
    end
end

uplo = CoreBlasLower
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
ref_coreblas_zgbtype1cb!(Float64, uplo, n, nb, A1, VQ1, TAUQ1, VP1, TAUP1, st, ed, sweep, Vblksiz, wantz)
display(A1)
display(VP1)
display(VQ1)
display(TAUP1)
display(TAUQ1)

Random.seed!(0)
A2 = rand(Float64, (3*nb+1, n))
VP2 = zeros(Float64, n)
VQ2 = zeros(Float64, n)
TAUP2 = zeros(Float64, n)
TAUQ2 = zeros(Float64, n)
coreblas_zgbtype1cb!(uplo, n, nb, A2, VQ2, TAUQ2, VP2, TAUP2, st, ed, sweep, Vblksiz, wantz)
# display(A2)
# display(VP2)
# display(VQ2)
# display(TAUP2)
# display(TAUQ2)

display(test_approx_equal(A1, A2))
display(test_approx_equal(VP1, VP2))
display(test_approx_equal(VQ1, VQ2))
display(test_approx_equal(TAUP1, TAUP2))
display(test_approx_equal(TAUQ1, TAUQ2))