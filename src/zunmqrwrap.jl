using LinearAlgebra 
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc

function unmqr!(::Type{T}, side::AbstractChar, trans::AbstractChar, 
    A::AbstractMatrix{T}, Tau::AbstractVector{T}, 
    C::AbstractMatrix{T})  where {T<: Number}

    mA, k = size(A)
    m,n = size(C)
    lda = max(1, stride(A,2))
    ldc = max(1, stride(C,2))
    work = Vector{T}(undef, 1)
    lwork = BlasInt(-1)
    info = Ref{BlasInt}()

    for i in 1:2
        if T == ComplexF64
            ccall((@blasfunc(zunmqr_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}),
            side, trans, m, n, k, A, lda, Tau, 
            C, ldc, work, lwork, info)
    
            if i == 1
                lwork = BlasInt(real(work[1]))
                resize!(work, lwork)
            end
    
            chklapackerror(info[])
        elseif T == Float64
            ccall((@blasfunc(dormqr_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}),
            side, trans, m, n, k, A, lda, Tau, 
            C, ldc, work, lwork, info)
    
            if i == 1
                lwork = BlasInt(real(work[1]))
                resize!(work, lwork)
            end
    
            chklapackerror(info[])        
        elseif T == ComplexF32
            ccall((@blasfunc(cunmqr_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}),
            side, trans, m, n, k, A, lda, Tau, 
            C, ldc, work, lwork, info)
    
            if i == 1
                lwork = BlasInt(real(work[1]))
                resize!(work, lwork)
            end
    
            chklapackerror(info[])
        else #T = Float32
            ccall((@blasfunc(sormqr_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}),
            side, trans, m, n, k, A, lda, Tau, 
            C, ldc, work, lwork, info)
    
            if i == 1
                lwork = BlasInt(real(work[1]))
                resize!(work, lwork)
            end
    
            chklapackerror(info[])
        end

    end
end
