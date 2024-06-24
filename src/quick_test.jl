using Random

n=4
nb=2
# Random.seed!(1234)
# A = rand(3*nb+1, n)
# VP = zeros(n)
# VQ = zeros(n)
# TAUP = zeros(n)
# TAUQ = zeros(n)
# ref_coreblas_zgbtype1cb!(Float64, n, nb, A, VQ, TAUQ, VP, TAUP)
# display(A)
# display(VP)
# display(VQ)
# display(TAUP)
# display(TAUQ)

Random.seed!(1234)
A = rand(Float64, (3*nb+1, n))
# A = ones(Float64, (3*nb+1, n))
# A[2,:] .= 2
# A[3,:] .= 3
VP = zeros(Float64, n)
VQ = zeros(Float64, n)
TAUP = zeros(Float64, n)
TAUQ = zeros(Float64, n)
display(A)
# display(VP)
# display(VQ)
# display(TAUP)
# display(TAUQ)
display("")
test_coreblas_zgbtype1cb!(Float64, n, nb, A, VQ, TAUQ, VP, TAUP)
display(A)
display(VP)
display(VQ)
display(TAUP)
display(TAUQ)