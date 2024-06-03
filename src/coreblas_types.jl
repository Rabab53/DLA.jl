#include <complex.h>

@enum COREBLAS_TYPE begin
    CoreBlasByte          = 0
    CoreBlasInteger       = 1
    CoreBlasRealFloat     = 2
    CoreBlasRealDouble    = 3
    CoreBlasComplexFloat  = 4
    CoreBlasComplexDouble = 5
end

@enum COREBLAS_OPT begin
    CoreBlasInvalid       = -1

    CoreBlasNoTrans       = 111
    CoreBlasTrans         = 112
    CoreBlasConjTrans     = 113
    # CoreBlas_ConjTrans    = CoreBlasConjTrans

    CoreBlasUpper         = 121
    CoreBlasLower         = 122
    CoreBlasGeneral       = 123
    CoreBlasGeneralBand   = 124

    CoreBlasNonUnit       = 131
    CoreBlasUnit          = 132

    CoreBlasLeft          = 141
    CoreBlasRight         = 142

    CoreBlasOneNorm       = 171
    CoreBlasRealOneNorm   = 172
    CoreBlasTwoNorm       = 173
    CoreBlasFrobeniusNorm = 174
    CoreBlasInfNorm       = 175
    CoreBlasRealInfNorm   = 176
    CoreBlasMaxNorm       = 177
    CoreBlasRealMaxNorm   = 178

    CoreBlasNoVec         = 301
    CoreBlasVec           = 302
    CoreBlasCount         = 303
    CoreBlasIVec          = 304
    CoreBlasAllVec        = 305
    CoreBlasSomeVec       = 306

    CoreBlasRangeAll      = 351
    CoreBlasRangeV        = 352
    CoreBlasRangeI        = 353

    CoreBlasForward       = 391
    CoreBlasBackward      = 392

    CoreBlasColumnwise    = 401
    CoreBlasRowwise       = 402

    CoreBlasW             = 501
    CoreBlasA2            = 502
    CoreBlas_Const_Limit  # Ensure always last.
end

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

const coreblas_enum_t = Int
const coreblas_complex32_t = ComplexF32
# const coreblas_complex64_t = ComplexF64
const coreblas_complex64_t = Float64

function coreblas_eigt_const(lapack_char::Char)::coreblas_enum_t end
function coreblas_job_const(lapack_char::Char)::coreblas_enum_t end
function coreblas_range_const(lapack_char::Char)::coreblas_enum_t end
function coreblas_diag_const(lapack_char::Char)::coreblas_enum_t end
function coreblas_direct_const(lapack_char::Char)::coreblas_enum_t end
function coreblas_norm_const(lapack_char::Char)::coreblas_enum_t end
function coreblas_side_const(lapack_char::Char)::coreblas_enum_t end
function coreblas_storev_const(lapack_char::Char)::coreblas_enum_t end
function coreblas_trans_const(lapack_char::Char)::coreblas_enum_t end
function coreblas_uplo_const(lapack_char::Char)::coreblas_enum_t end