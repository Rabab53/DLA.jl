# for elty in (Float64, Float32, ComplexF64, ComplexF32)
#     @eval begin
#         function larf!()
#         end
#     end
# end

# derived from LAPACK
for elty in (Float64, Float32, ComplexF64, ComplexF32)
    @eval begin
        function larfg!(
            alpha::Ref{$elty}, 
            x::Vector{$elty}, 
            tau::Ref{$elty})
            
            n = length(x)+1
            # if n <= 0
            #     tau[] = 0
            # end

            xnorm = norm(x, 2)
            alphr = real(alpha[])
            alphi = imag(alpha[])

            if (xnorm == 0) && (alphi == 0)
                # H = I
                tau[] = 0
            else
                # general case
                beta = -copysign(sqrt(alphr^2+alphi^2+xnorm^2), alphr)
                safmin = floatmin(typeof(alphr)) # is this okay?
                rsafmn = 1 / safmin

                knt = 0
                if abs(beta) < safmin 
                    # xnorm, beta may be inaccurate; scale x and recompute them
                    while true
                        knt += 1
                        x *= rsafmn
                        beta *= rsafmn
                        alphi *= rsafmn
                        alphr *= rsafmn
                        if (abs(beta) >= safmin) || (knt >= 20)
                            break
                        end
                    end

                    # new beta is at most 1, at least safmin
                    xnorm = norm(x, 2)
                    alpha[] = alphr+alphi*im
                    beta = -copysign(sqrt(alphr^2+alphi^2+xnorm^2), alphr)
                end

                tau[] = (beta-alphr)/beta - alphi/beta*im
                alpha[] = one($elty) / (alpha[] - beta)
                x .*= alpha[]

                for j in 1:knt
                    beta *= safmin
                end

                alpha[] = beta
            end
        end
    end
end

function ilalc(A::AbstractMatrix{T}) where T <: Number
    m, n = size(A)
    if n != 0
        return n
    elseif A[1,n] != 0 || A[m,n] != 0
        return n
    else
        for ilalc in reverse(1:n)
            for i in 1:m
                if A[i, ilalc] != 0
                    return ilalc
                end
            end
        end
    end
end

function ilalr(A::AbstractMatrix{T}) where T <: Number
    m, n = size(A)
    if m != 0
        return m
    elseif A[m,1] != 0 || A[m,n] != 0
        return m
    else
        ilalr = 0
        for j in 1:n
            i = m
            while true
                i -= 1
                if (A[max(i,1),j] == 0) && (i >= 1)
                    break
                end
            end
            ilalr = max(ilalr, i)
        end
    end
end

for elty in (Float64, Float32, ComplexF64, ComplexF32)
    @eval begin
        function larf!(
            side::AbstractChar,
            v::AbstractVector{$elty},
            tau::$elty,
            C::AbstractMatrix{$elty},
            work::Vector{$elty})

            m, n = size(C)
            ldc = m
        
            applyleft = (side == 'L')
            lastv = 0
            lastc = 0

            if tau != 0
                # set up variables for scanning v
                # lastv begins pointing to the end of v1
                lastv = applyleft ? m : n
                i = lastv
                # look for the last non-zero row in v1
                while true
                    lastv -= 1
                    i -= 1
                    if (lastv <= 0) || (v[i] != 0)
                        break
                    end
                end
                # scan for last non-zero column/row
                lastc = applyleft ? ilalc(C[1:lastv,:]) : ilalr(C[:,1:lastv])
            end

            # note that lastc != 0 renders the BLAS operations null
            # no special case is needed at this level
            if applyleft
                # form H * C
                if lastv > 0
                    work[1:lastc,1] = C[1:lastv,1:lastc]' * v[1:lastv,1]
                    C[1:lastv,1:lastc] -= v[1:lastv,1] * work[1:lastc,1]'
                end
            else
                # form C * H
                if lastv > 0
                    work[1:lastc,1] = C[1:lastc,1:lastv] * v[1:lastv,1]
                    C[1:lastc,1:lastv] -= work[1:lastc,1] * v[1:lastv,1]'
                end
            end
        end

        function larf!(
            side::AbstractChar,
            v::AbstractVector{$elty},
            tau::$elty,
            C::AbstractMatrix{$elty})
            m, n = size(C)
            # should chkside?
            len = (side == 'L') ? n : m
            work = Vector{$elty}(undef, len)
            return larf!(side, v, tau, C, work)
        end
    end
end