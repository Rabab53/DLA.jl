module DLA

using Base: require_one_based_indexing, USE_BLAS64

using LinearAlgebra: LAPACK

include("bulge.jl")

include("coreblas_types.jl")

include("core_zgbtype1cb.jl")

include("ref_core_zgbtype1cb.jl")

include("quick_test.jl")

end
