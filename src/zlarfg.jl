using LinearAlgebra

function zlarfg(n, alpha, x, incx, tau)
    one = oneunit(eltype(alpha))
    zero0 = zero(eltype(alpha)) 

    if n <= 0
        tau = zero0
        return alpha, tau
    end
    
    if n == 1
        xnorm = 0
    else
        xnorm = norm(x,2)
    end

    alphr = real(alpha)
    alphi = imag(alpha)

    if xnorm == 0 && alphi == 0
        tau = zero0
    else
        beta = -copysign(sqrt(alphr^2 + alphi^2 + xnorm^2), alphr)
        safmin = lamch(eltype(alphr), 'S') / lamch(eltype(alphr), 'E')
        rsafmn = one / safmin
        knt = 0

        if abs(beta) < safmin
            #  xnorm, beta may be inaccurate, scale x and recompute
            
            while true
                knt += 1
                # change to internal
                #LinearAlgebra.rmul!(x, rsafmn)
                #LinearAlgebra.generic_mul!(x, rsafmn, x, LinearAlgebra.MulAddMul(one, zero0))
                x .*= rsafmn
                beta *= rsafmn
                alphr *= rsafmn
                alphi *= rsafmn
                alpha *= rsafmn

                if abs(beta) < safmin
                    break
                end
            end                

            #recompute 
            xnorm = norm(x)
            beta = -copysign(sqrt(alphr^2 + alphi^2 + xnorm^2), alphr)
        end

        tau = (beta - alpha) / beta
        #LinearAlgebra.rmul!(x, (one / (alpha-beta)))
        #LinearAlgebra.generic_mul!(x, one / (alpha-beta), x, LinearAlgebra.MulAddMul(one, zero0))
        x .*= (one / (alpha-beta))

        for j in 1:knt
            beta *= safmin
        end

        alpha = beta
    end

    return alpha, tau
end

# currently only handles float related procedures
function lamch(::Type{T}, cmach) where{T<: Number}
    ep = eps(T)

    if cmach == 'E'
        return ep
    end

    #assume cmach = 'S'

    one = oneunit(T)
    sfmin = floatmin(T)
    small = one / floatmax(T)

    if small >= sfmin
        return small*(one + ep)
    end

    return sfmin
end