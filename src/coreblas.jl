module CoreBLAS

using Base: require_one_based_indexing, USE_BLAS64

using LinearAlgebra: LAPACK

include("bulge.jl")
using .Bulge

include("coreblas_types.jl")
using .CoreBlasTypes 

include("core_zgbtype1cb.jl")

end