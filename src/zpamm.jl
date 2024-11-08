using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools

function zpamm(op, side, storev, m, n, k, l, A1, lda1, A2, lda2, V, ldv, W, ldw)
    if op != 'W' && op != 'A'
        throw(ArgumentError("illegal value of op"))
        return -1
    end

    if side != 'L' && side != 'R'
        throw(ArgumentError("illegal value of side"))
        return -2
    end

    if storev!= 'C' && storev != 'R'
        throw("illegal value o storev")
        return -3
    end

    if m < 0
        throw(ArgumentError("illegal value of m"))
        return -4
    end

    if n < 0
        throw(ArgumentError("illegal value of n"))
        return -5
    end

    if k < 0
        throw(ArgumentError("illegal value of k"))
        return -6
    end

    if l < 0
        throw(ArgumentError("illegal value of l"))
        return -7
    end

    if lda1 < 0
        throw(ArgumentError("illegal value of lda1"))
        return -9
    end

    if lda2 < 0
        throw(ArgumentError("illegal value of lda2"))
        return -11
    end

    if ldv < 0
        throw(ArgumentError("illegal value of ldv"))
        return -13
    end

    if ldw < 0
        throw(ArgumentError("illegal value of ldw"))
        return -15
    end

    #quick return 
    if m == 0 || n == 0 || k == 0
        return 
    end

    if storev == 'C'
        uplo = 'U'

        if side == 'L'
            if op == 'A'
                trans = 'N'
            else
                trans = 'C'
            end
        else
            if op == 'W'
                trans = 'N'
            else
                trans = 'C'
            end
        end
    else
        uplo = 'L'
        if side == 'L'
            if op == 'W'
                trans = 'N'
            else
                trans = 'C'
            end
        else
            if op == 'A'
                trans = 'N'
            else
                trans = 'C'
            end
        end
    end

    if op == 'W'
        zpamm_w(side, trans, uplo, m,n,k,l, A1, A2, V, W)
    else
        zpamm_a(side, trans, uplo, m,n,k,l, A2, V, W)
    end
    
    return 
end

function zpamm_w(side, trans, uplo, m, n, k, l, A1, A2, V,W)
    # W = A1 + op(V) * A2 or W = A1 + A2 * op(V)
    one0 = oneunit(eltype(A1))
    zero0 = zero(eltype(A1))
    plus = LinearAlgebra.MulAddMul(one0, one0)
    eqa = LinearAlgebra.MulAddMul(one0, zero0)

    if side == 'L'
        if trans == 'C' && uplo == 'U'
            # W = A1 + V^H * A2
            
            #W = A2_2
            copyto!(W, CartesianIndices((1:l, 1:n)), A2, CartesianIndices((k-l+1:k, 1:n)))

            # W = V_2 ^H * W  + V_1^H * A2_1 (top l rows of V^H)
            if l > 0
                #W = V_2 ^ H * W
                LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), uplo, 'N', adjoint,  (@view V[k-l+1:k, 1:l]),  (@view W[1:l, 1:n]))

                # W = W + V1^H + A2_1
                if k > l
                    LinearAlgebra.generic_matmatmul!((@view W[1:l, 1:n]), trans, 'N', (@view V[1:k-l, 1:l]), (@view A2[1:k-l, 1:n]), plus)
                end
            end

            # W_2 = V_3^H * A2 (ge, bottom m-l rows of v^H)
            if m > l
                LinearAlgebra.generic_matmatmul!((@view W[l+1:m, 1:n]), trans, 'N', (@view V[1:k, l+1:m]), (@view A2[1:k, 1:n]), eqa)
            end
            # W = A1 + w
            for j in 1:n
                LinearAlgebra.axpy!(one0, (@view A1[1:m, j]), (@view W[1:m, j]))
            end

        elseif trans == 'N' && uplo == 'L'
            # W = A1 + V^H * A2
            #W = A2_2
            copyto!(W, CartesianIndices((1:l, 1:n)), A2, CartesianIndices((k-l+1:k, 1:n)))
            
            # W = V_2 ^H * W  + V_1^H * A2_1 (top l rows of V^H)

            if l > 0
                #W = V_2 ^ H * W
                LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), uplo, 'N', identity,  (@view V[1:l, k-l+1:k]),  (@view W[1:l, 1:n]))

                # W = W + V1^H + A2_1
                if k > l
                    LinearAlgebra.generic_matmatmul!((@view W[1:l, 1:n]), trans, 'N', (@view V[1:l, 1:k-l]), (@view A2[1:k-l, 1:n]), plus)
                end
            end

            # W_2 = V_3^H * A2 (ge, bottom m-l rows of v^H)
            if m > l
                LinearAlgebra.generic_matmatmul!((@view W[l+1:m, 1:n]), trans, 'N', (@view V[1:k, l+1:m]), (@view A2[1:k, 1:n]), eqa)
            end
            # W = A1 + w
            
            for j in 1:n
                LinearAlgebra.axpy!(one0, (@view A1[1:m, j]), (@view W[1:m, j]))
            end

        else
            throw(ErrorException("not yet supported"))
            return 
        end
    else #side == 'R' 
        if (trans == 'C' && uplo == 'U') || (trans == 'N' && uplo == 'L')
            throw(ErrorException("not yet supported"))
            return
        elseif (trans == 'N' && uplo == 'U')
            #W = a1 + A2 * V

            if l > 0
                # W = A2_2
                copyto!(W, CartesianIndices((1:m, 1:l)), A2, CartesianIndices((1:m, k-l+1:k)))

                #W = W * V2 --> W = A2_2 * V2
                LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), uplo, 'N', identity, (@view W[1:m, 1:l]),  (@view V[k-l+1:k, 1:l]))

                # W = W + A2_1 * V_1
                if k > l
                    LinearAlgebra.generic_matmatmul!((@view W[1:m, 1:l]), 'N', trans, (@view A2[1:m, 1:k-l]), (@view V[1:k-l, 1:l]), plus)
                end
            end

            #W = W + A2 * V_3
            if n > l
                LinearAlgebra.generic_matmatmul!((@view W[1:m, l+1:n]), 'N', trans,(@view A2[1:m, 1:k]), (@view V[1:k, l+1:n]), eqa)
            end

            # W = A1 + W

            for j in 1:n
                LinearAlgebra.axpy!(one0, (@view A1[1:m, j]), (@view W[1:m, j]))
            end

        else # uplo = 'L' trans == C

             #W = a1 + A2 * V
             if l > 0

                # W = A2_2
                copyto!(W, CartesianIndices((1:m, 1:l)), A2, CartesianIndices((1:m, k-l+1:k)))

                #W = W * V2 --> W = A2_2 * V2
                LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), uplo, 'N', adjoint, (@view W[1:m, 1:l]),  (@view V[1:l, k-l+1:k]))

                # W = W + A2_1 * V_1
                if k > l
                    LinearAlgebra.generic_matmatmul!((@view W[1:m, 1:l]), 'N', trans,  (@view A2[1:m, 1:k-l]), (@view V[1:l, 1:k-l]), plus)
                end
            end

            #W = W + A2 * V_3
            if n > l
                LinearAlgebra.generic_matmatmul!((@view W[1:m, l+1:n]), 'N', trans,(@view A2[1:m, 1:k]), (@view V[l+1:n, 1:k]), eqa)
            end

            # W = A1 + W
            for j in 1:n
                LinearAlgebra.axpy!(one0, (@view A1[1:m, j]), (@view W[1:m, j]))
            end
        end
    end

    return
end

function zpamm_a(side, trans, uplo, m, n, k, l, A2, V, W)
    # A2 = A2 + op(V) * W or A2 = A2 + W * op(V)
    one0 = oneunit(eltype(A2))
    plus = LinearAlgebra.MulAddMul(one0, one0)
    minus = LinearAlgebra.MulAddMul(one0*(-1),one0)

    if side == 'L'
        if (trans == 'C' && uplo == 'U') || (trans == 'N' && uplo == 'L')
            throw(ErrorException("not yet implmented"))
            return 
        elseif (trans == 'C' && uplo == 'L')
            # A2 = A2 - V * w
            # A2_1 = A2_1 - V_1 * W_1

            if m > l
                LinearAlgebra.generic_matmatmul!((@view A2[1:m-l, 1:n]), trans, 'N', (@view V[1:l, 1:m-l]), (@view W[1:l, 1:n]), minus)
            end

            #W_1 = V_2 * W_1
            LinearAlgebra.generic_trimatmul!( (@view W[1:l, 1:n]), uplo, 'N', adjoint, (@view V[1:l, m-l+1:m]),   (@view W[1:l, 1:n]))
            
            #A2_2 = A2_2 - W_1
            for j in 1:n
                LinearAlgebra.axpy!(-one0, (@view W[1:l, j]), (@view A2[m-l+1:m, j]))
            end

            # A2 = A2 - V_3 * W_2
            if k > l
                LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:n]), trans, 'N',(@view V[l+1:k, 1:m]), (@view W[l+1:k, 1:n]), minus)
            end

        else # (trans == 'N' && uplo == 'U')
            # A2 = A2 - V * w
            # A2_1 = A2_1 - V_1 * W_1

            if m > l
                LinearAlgebra.generic_matmatmul!((@view A2[1:m-l, 1:n]), trans, 'N', (@view V[1:m-l, 1:l]), (@view W[1:l, 1:n]), minus)
            end

            #W_1 = V_2 * W_1
            LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), uplo, 'N', identity, (@view V[m-l+1:m, 1:l]), (@view W[1:l, 1:n]))
            
            #A2_2 = A2_2 - W_1

            for j in 1:n
                LinearAlgebra.axpy!(-one0, (@view W[1:l, j]), (@view A2[m-l+1:m, j]))
            end

            # A2 = A2 - V_3 * W_2
            if k > l
                LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:n]), trans, 'N', (@view V[1:m, l+1:k]), (@view W[l+1:k, 1:n]), minus)
            end

        end
    else # side = 'R'
        if (trans == 'C' && uplo == 'U')
            #A2 = A2 - W* V^H
            #A2 = A2 - W_2 * V_3^H

            if k > l
                LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:n]), 'N', trans, (@view W[1:m, l+1:k]), (@view V[1:n, l+1:k]), minus)
            end

            #A2_1 = A2_1 - W-1 * V_1^H
            if n > l
                LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:n-l]), 'N', trans, (@view W[1:m, 1:l]), (@view V[1:n-l, 1:l]), minus)
            end

            #A2_2 = A2_2 - W_1 * V_2^H
            if l > 0
               # LinearAlgebra.BLAS.trmm!('R', uplo, trans, 'N', -one0, (@view V[n-l+1:n, 1:l]), (@view W[1:m, 1:l]))
                LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), uplo, 'N', adjoint, (@view W[1:m, 1:l]), -(@view V[n-l+1:n, 1:l]))
            end

            for j in 1:l
                LinearAlgebra.axpy!(one0, (@view W[1:m, j]), (@view A2[1:m, n-l+j]))
            end

        elseif  (trans == 'N' && uplo == 'L')

            if k > l
                LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:n]), 'N', trans, (@view W[1:m, l+1:k]), (@view V[l+1:k, 1:n]), minus)
            end

            #A2_1 = A2_1 - W-1 * V_1^H
            if n > l
                LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:n-l]), 'N', trans, (@view W[1:m, 1:l]), (@view V[1:l, 1:n-l]), minus)
            end

            #A2_2 = A2_2 - W_1 * V_2^H
            if l > 0
                #LinearAlgebra.BLAS.trmm!('R', uplo, trans, 'N', -one0, (@view V[1:l, n-l+1:n]), (@view W[1:m, 1:l]))
                LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), uplo, 'N', identity, (@view W[1:m, 1:l]), -(@view V[1:l, n-l+1:n]))
            end

            for j in 1:l
                LinearAlgebra.axpy!(one0, (@view W[1:m, j]), (@view A2[1:m, n-l+j]))
            end
        else
            throw(ErrorException("not yet implemented"))
            return
        end
    end

    return
end

