module DLA

using Base: require_one_based_indexing, USE_BLAS64

import LinearAlgebra
using LinearAlgebra

include("bulge.jl")

include("lapack_alt.jl")
include("lapack_native.jl")

include("coreblas_types.jl")

include("coreblas_gbtype1cb.jl")
include("coreblas_gbtype2cb.jl")
include("coreblas_gbtype3cb.jl")

include("ref_coreblas.jl")

# include("quick_test.jl")

end
