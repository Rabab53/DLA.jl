using LinearAlgebra

function lulu!(A::Matrix{T}) where T
    m, n = size(A)
    LDA = size(A, 1)
    ipiv = zeros(Int, min(m, n))
    info = Ref{Int}(0)

    if m < 0
        info[] = -1
        return
    end
    if n < 0
        info[] = -2
        return
    end
    if LDA < max(1,m)
        info[] = -4
        return
    end

    # quick return
    if m == 0 || n == 0
        return
    end

    # implmenting the unblocked algorithm
    A, ipiv, info[] = getrf2!(A, ipiv, info)
    return A, ipiv, info[]
end

function getrf2!(A::Matrix{T}, ipiv::Vector{Int}, info::Ref{Int}) where T
    m, n = size(A)
    lda = m
    info[] = 0
   # iinfo[] = Ref{Int}(0)

    if m < 0
        info[] = -1
        return
    end
    if n < 0
        info[] = -2
        return
    end
    if lda < max(1,m)
        info[] = -4
        return
    end

    # quick return
    if m == 0 || n == 0
        return
    end

    if m == 1
        ipiv[1] = 1
        if A[1,1] == zero(T)
            info[] = 1
            return
        end
    elseif n == 1
        sfmin = eps(T)
        dmax = abs(A[1,1])
        for i = 2:m
            if abs(A[i,1]) > dmax
                idamax = i
                dmax = abs(A[i,1])
            end
        end
        ipiv[1] = idamax
        if A[idamax,1] != zero(T)
            #Apply the interchange
            if idamax != 1
                temp = A[1,1]
                A[1,1] = A[idamax,1]
                A[idamax,1] = temp
            end
            #Compute element 2:m of the column
            if abs(A[1, 1]) >= sfmin
                BLAS.scal(m-1, one(T)/A[1,1], view(A, 2:m, 1), 1)
            else
                for i = 1:m-1
                    A[i+1, 1] = A[i+1, 1] / A[1, 1]
                end
            end
        else
            info[] = 1
            return
        end
    else
        #use recursive code
        n1 = min(m, n) /2
        n2 = n - n1

        #          [A11]
        #Factor    [---]
        #          [A12]
       # getrf2!(A, ipiv, iinfo)
       # if info[] != 0 && iinfo[] > 0
       #     info[] = iinfo[] 
       # end

        # Apply interchanges to [A12]
        #                       [---]
        #                       [A22]

        laswp(view(A, :, n1+1:n), Int(1), Int(n1), ipiv, Int(1))

        # Solve A12
      #  BLAS.trsm!('L', 'L', 'N', 'U', one(T), A, view(A, :, n1+1:n))
      B = view(A, :, n1+1:n)
      L = LowerTriangular(A)
      #L[diag(L)] = one(T)
      L[diagind(L)] .= one(T)
      X = L \ B
      A[:, n1+1:n] .= X



        # Update A22
       # BLAS.gemm!('N', 'N', -one(T), view(A, n1+1:m, :), view(A, :, n1+1:n), one(T), view(A, n1+1:m, n1+1:n))

        #Factor A22
      #  getrf2!(view(A, n1+1:m, n1+1:n), ipiv(n1+1:min(m,n)), iinfo)
"""
        if info[] == 0 && iinfo[] > 0
            info[] = iinfo[] + n1
            for i = n1+1:min(m,n)
                ipiv[i] += n1
            end
        end
"""
        laswp(view(A, :, :),  Int(n1+1), Int(min(m,n)), ipiv, Int(1))
    end

    return A, ipiv, info[]
end

function laswp(A::AbstractMatrix{T}, first::Integer, last::Integer, ipiv::AbstractVector{Int}, incx::Integer) where T
    m, n = size(A)
    k1 = first
    k2 = last
    if incx > 0
        ix0 = k1
        i1 = k1
        i2 = k2
        inc = Int(1)
    elseif incx < 0
        ix0 = 1+ (1 - k2) * incx
        i1 = k2
        i2 = k1
        inc = Int(-1)
    else
        return
    end

    n32 = (n ÷ 32) * 32

    if n32 != 0
        for j in 1:32:n32
            @show j, typeof(j)
            ix = ix0
            for i in i1:inc:i2
                ip = ipiv[ix]
                if ip != i
                    for k in j:j+31
                        @show i, k, ip, j, typeof(i), typeof(k), typeof(ip), typeof(j)
                        temp = A[i, k]
                        A[i, k] = A[ip, k]
                        A[ip, k] = temp
                    end
                end
                ix += inc
            end
        end
    end
    if n32 != n
        n32 +=1
        ix = ix0
        for i in i1:inc:i2
            ip = ipiv[ix]
            if ip != i
                for k in n32:n
                    temp = A[i, k]
                    A[i, k] = A[ip, k]
                    A[ip, k] = temp
                end
            end
            ix += inc
        end
    end
    return
end
"""
mm = mod(m-1,5)
scale = one(T)/A[1,1]
if mm != 0
    for i = 1:mm
        A[i, 1] *= scale
    end
end
if m >= 5
    mp1 = m - 5
    for i = mp1:5:m
        A[i, 1] *= scale
        A[i+1, 1] *= scale
        A[i+2, 1] *= scale
        A[i+3, 1] *= scale
        A[i+4, 1] *= scale
    end
end
"""
