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