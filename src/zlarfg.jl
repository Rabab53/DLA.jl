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
        x .*= (one / (alpha-beta))

        for j in 1:knt
            beta *= safmin
        end

        alpha = beta
    end

    return alpha, tau
end

"""
    lamch(::Type{T}, cmach) where{T<: Number}

Determines single / double precision machine parameters

# Arguments
- T : type, currently only tested Float32 and Float64
- 'cmach' : specifies the value to be returned by lamch
    - = 'E': returns eps
    - = 'S': returns sfmin
    - = 'P': returns eps*base
    
    - where
        - eps = relative machine precision
        - sfmin = safe min, such that 1/sfmin does not overflow
        - base = base of the machine
"""
function lamch(::Type{T}, cmach) where{T<: Number}
    ep = eps(T) 
    one = oneunit(T)
    rnd = one

    if one == rnd
        ep *= 0.5
    end

    if cmach == 'E'
        return ep
    elseif cmach == 'S'
        sfmin = floatmin(T)
        small = one / floatmax(T)

        if small >= sfmin
            sfmin = small*(one + ep)
        end
        return sfmin
    else # assume cmach = 'P'
        # assume base of machine is 2
        return ep*2
    end
end