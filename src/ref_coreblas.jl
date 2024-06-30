using LinearAlgebra, libcoreblas_jll, OpenBLAS32

macro coreblasfunc(x)
    return Expr(:quote, Symbol("coreblas_", x))
end

macro coreblasfname(x)
    return Symbol("ref_coreblas_", x, "!")
end

# function @coreblasfname(:clag2z)(
#     As::AbstractMatrix{ComplexF32},
#     A::AbstractMatrix{ComplexF64})

#     @assert size(As) == size(A)
#     m, n = size(As)
#     ldas = size(As, 1)
#     lda = size(A, 1)

#     ccall((:coreblas_clag2z, "libcoreblas.so"), Cvoid,
#         (Int64, Int64, Ptr{ComplexF32}, Int64, Ptr{ComplexF64}, Int64),
#         m, n, As, ldas, A, lda)
# end

for (fname, gbtypexcb, elty) in 
    ((:gbtype1cb, :dgbtype1cb, Float64), 
     (:gbtype1cb, :sgbtype1cb, Float32), 
     (:gbtype1cb, :zgbtype1cb, ComplexF64),
     (:gbtype1cb, :cgbtype1cb, ComplexF32),
     (:gbtype2cb, :dgbtype2cb, Float64), 
     (:gbtype2cb, :sgbtype2cb, Float32), 
     (:gbtype2cb, :zgbtype2cb, ComplexF64),
     (:gbtype2cb, :cgbtype2cb, ComplexF32),
     (:gbtype3cb, :dgbtype3cb, Float64), 
     (:gbtype3cb, :sgbtype3cb, Float32), 
     (:gbtype3cb, :zgbtype3cb, ComplexF64),
     (:gbtype3cb, :cgbtype3cb, ComplexF32))
    @eval begin
        function @coreblasfname($fname)(
            uplo,
            N, 
            nb, 
            A::AbstractMatrix{$elty}, 
            VQ::Vector{$elty},  
            TAUQ::Vector{$elty}, 
            VP::Vector{$elty},  
            TAUP::Vector{$elty},
            st,
            ed,
            sweep,
            Vblksiz,
            wantz)

            lda = size(A, 1)
            WORK = Vector{$elty}(undef, nb)

            ccall((@coreblasfunc($gbtypexcb), "libcoreblas.so"), Cvoid,
                    (Int64, Int64, Int64, Ptr{$elty}, Int64, Ptr{$elty}, Ptr{$elty},
                    Ptr{$elty}, Ptr{$elty}, Int64, Int64, Int64, Int64, Int64, Ptr{$elty}),
                    uplo, N, nb, A, lda, VQ, TAUQ, VP, TAUP, st, ed, sweep, Vblksiz, wantz, WORK)
        end
    end
end