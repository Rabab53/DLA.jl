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
A = rand(3*nb+1, n)
VP = zeros(n)
VQ = zeros(n)
TAUP = zeros(n)
TAUQ = zeros(n)
test_coreblas_zgbtype1cb!(Float64, n, nb, A, VQ, TAUQ, VP, TAUP)
display(A)
display(VP)
display(VQ)
display(TAUP)
display(TAUQ)