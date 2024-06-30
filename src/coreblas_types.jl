# @enum COREBLAS_TYPE begin
const CoreBlasByte          = 0
const CoreBlasInteger       = 1
const CoreBlasRealFloat     = 2
const CoreBlasRealDouble    = 3
const CoreBlasComplexFloat  = 4
const CoreBlasComplexDouble = 5
# end

# @enum COREBLAS_OPT begin
const CoreBlasInvalid       = -1

const CoreBlasNoTrans       = 111
const CoreBlasTrans         = 112
const CoreBlasConjTrans     = 113
const CoreBlas_ConjTrans    = CoreBlasConjTrans

const CoreBlasUpper         = 121
const CoreBlasLower         = 122
const CoreBlasGeneral       = 123
const CoreBlasGeneralBand   = 124

const CoreBlasNonUnit       = 131
const CoreBlasUnit          = 132

const CoreBlasLeft          = 141
const CoreBlasRight         = 142

const CoreBlasOneNorm       = 171
const CoreBlasRealOneNorm   = 172
const CoreBlasTwoNorm       = 173
const CoreBlasFrobeniusNorm = 174
const CoreBlasInfNorm       = 175
const CoreBlasRealInfNorm   = 176
const CoreBlasMaxNorm       = 177
const CoreBlasRealMaxNorm   = 178

const CoreBlasNoVec         = 301
const CoreBlasVec           = 302
const CoreBlasCount         = 303
const CoreBlasIVec          = 304
const CoreBlasAllVec        = 305
const CoreBlasSomeVec       = 306

const CoreBlasRangeAll      = 351
const CoreBlasRangeV        = 352
const CoreBlasRangeI        = 353

const CoreBlasForward       = 391
const CoreBlasBackward      = 392

const CoreBlasColumnwise    = 401
const CoreBlasRowwise       = 402

const CoreBlasW             = 501
const CoreBlasA2            = 502
const CoreBlas_Const_Limit  = 503 # Ensure always last.
# end

@enum COREBLAS_ERRORS begin
    CoreBlasSuccess = 0
    CoreBlasErrorNotInitialized
    CoreBlasErrorNotSupported
    CoreBlasErrorIllegalValue
    CoreBlasErrorOutOfMemory
    CoreBlasErrorNullParameter
    CoreBlasErrorInternal
    CoreBlasErrorSequence
    CoreBlasErrorComponent
    CoreBlasErrorEnvironment
end

@enum COREBLAS_PLACE begin
    CoreBlasInplace
    CoreBlasOutplace
end

@enum COREBLAS_HOUSEHOLDER begin
    CoreBlasFlatHouseholder
    CoreBlasTreeHouseholder
end

@enum COREBLAS_ENABLE begin
    CoreBlasDisabled = 0
    CoreBlasEnabled = 1
end

@enum COREBLAS_OPTION begin
    CoreBlasTuning
    CoreBlasNb
    CoreBlasIb
    CoreBlasInplaceOutplace
    CoreBlasNumPanelThreads
    CoreBlasHouseholderMode
end

# const coreblas_enum_t = Int
# const coreblas_complex32_t = ComplexF32
# const coreblas_complex64_t = ComplexF64
# const coreblas_complex64_t = Float64

# function coreblas_eigt_const(lapack_char::Char)::coreblas_enum_t end
# function coreblas_job_const(lapack_char::Char)::coreblas_enum_t end
# function coreblas_range_const(lapack_char::Char)::coreblas_enum_t end
# function coreblas_diag_const(lapack_char::Char)::coreblas_enum_t end
# function coreblas_direct_const(lapack_char::Char)::coreblas_enum_t end
# function coreblas_norm_const(lapack_char::Char)::coreblas_enum_t end
# function coreblas_side_const(lapack_char::Char)::coreblas_enum_t end
# function coreblas_storev_const(lapack_char::Char)::coreblas_enum_t end
# function coreblas_trans_const(lapack_char::Char)::coreblas_enum_t end
# function coreblas_uplo_const(lapack_char::Char)::coreblas_enum_t end