using LinearAlgebra

function zlarfbv0(side, trans, direct, storev, m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork)
    if m <= 0 || n <= 0
        return
    end

    if storev == 'C'
        if direct == 'F'
            if side == 'L'
                c1 = @view c[1:k,:]
                c2 = @view c[k+1:m,:]
                v1 = @view v[1:k,:] #unit lower triangular
                v2 = @view v[k+1:m,:]
            
                # W = C1^H
                # og copied and conjugated by row w/ zcopy
                for i in 1:k
                    for j in 1:n
                        work[j,i] = conj(c[i,j])
                    end
                end

                # W = W*V1, W = ldw by k, V1 is k by k
                # og ztrmm 
                work *= v1
                
                if m > k
                    # W = W + C2^H * V2
                    #og called zgemm
                    work .+= (c2')*(v2)
                end
                
                # W = W * T^H or W*T
                # call ztrmm

                if trans == 'N' # W = W*T^H
                    work *= (t')
                else
                    work *= (t)
                end

                if m > k #C2 = C2 - V2*W^H
                    #call zgemm
                    c2 .-= v2*(work')
                end

                # w = w*v1^H
                #call ztrmm
                work *= (v1')

                # c1 = c1 - w^h

                for j in 1:k
                    for i in 1:n
                        c[j,i] = c[j,i] - conj(work[i,j])
                    end
                end

            else 
                if side == 'R' || side == 'r'
                    c1 = @view c[:, 1:k]
                    c2 = @view c[:, k+1:n]
                    v1 = @view v[1:k,:]
                    v2 = @view v[k+1:n,:]

                    #W = C1
                    for i in 1:m
                        for j in 1:k
                            work[i,j] = c[i,j]
                        end
                    end

                    # w = w*v1
                    #call ztrmm
                    work = work*v1

                    if n > k
                        #call zgemm
                        # w = w + c2*v2
                        work .+= c2*v2
                    end
                    
                    #w = w*t or w*t^H

                    if trans == 'C' # W = W*T^H
                        work = work*(t')
                    else
                        work = work*t
                    end

                    if n > k
                        # c2 = c2 - w*v2^H
                        c2 .-= work*(v2')
                    end

                    #call ztrmm
                    work = work*(v1')

                    #c1 = c1 - w
                    for j in 1:k
                        for i in 1:m
                            c[i,j] = c[i,j] - work[i,j]
                       end
                    end
                end
            end
        else

            if side == 'L' || side == 'l'
                c1 = @view c[1:m-k,:]
                c2 = @view c[m-k+1:m,:]
                v1 = @view v[1:ldv-k,:]
                v2 = @view v[ldv-k+1:ldv,:]
                
                # w = c2^h
                for i in 1:k
                    for j in 1:n
                        work[j,i] = conj(c2[i,j])
                    end
                end

                work = work*v2

                if m > k
                    #work = work + (c1')*v1
                    work .+= (c1')*v1
                end

                if trans == 'N'
                    work = work*(t')
                else
                    work = work*t
                end

                #c1 = c1 - v1*w^H
                if m > k
                    c1 .-= v1*(work')
                end

                work = work*(v2')

                #c2 = c2 - w^H
                for j in 1:k
                    for i in 1:n
                        c[m-k+j,i] = c[m-k+j,i] - conj(work[i,j])
                    end
                end
            else 
                if side == 'R' || side == 'r'
                    c1 = @view c[:,1:n-k]
                    c2 = @view c[:,n-k+1:n]
                    v1 = @view v[1:ldv-k,:]
                    v2 = @view v[ldv-k+1:ldv,:]

                    # w = c2
                    for i in 1:m
                        for j in 1:k
                            work[i,j] = c2[i,j]
                        end
                    end

                    work = work*v2

                    if n > k
                        #work = work + c1*v1
                        work .+= c1*v1
                    end

                    if trans == 'C'
                        work = work*(t')
                    else
                        work = work*t
                    end
                    
                    #c1 = c1 - w*v1^H
                    if n > k
                        c1 .-= work*(v1')
                    end

                    work = work*(v2')

                    #c2 = c2-w
                    for j in 1:k
                        for i in 1:m
                            c[i,n-k+j] = c[i,n-k+j] - work[i,j]
                        end
                    end
                end
            end
        end
    else 
        if storev == 'R'
            if direct == 'F'
                if side == 'L'
                    v1 = @view v[:, 1:k]
                    v2 = @view v[:, k+1:m]
                    c1 = @view c[1:k, :]
                    c2 = @view c[k+1:m, :]

                    #w = c1^h
                    for i in 1:k
                        for j in 1:n
                            work[j,i] = conj(c1[i,j])
                        end
                    end

                    work = work*(v1')

                    if m > k
                        #work = work + (c2')*(v2')
                        work .+= (c2')*(v2')
                    end

                    if trans == 'N'
                        work = work*(t')
                    else
                        work = work*t
                    end

                    #c2 = c2 - v2^h*w^h
                    if m > k
                        c2 .-= (v2')*(work')
                    end

                    work = work*v1

                    #c1 = c1 - w^h
                    for j in 1:k
                        for i in 1:n
                            c[j,i] = c[j,i] - conj(work[i,j])
                        end
                    end

                else 
                    if side == 'R' || side == 'r'
                        v1 = @view v[:, 1:k]
                        v2 = @view v[:, k+1:n]
                        c1 = @view c[:, 1:k]
                        c2 = @view c[:, k+1:n]

                        #w = c1
                        for i in 1:m
                            for j in 1:k
                                work[i,j] = c1[i,j]
                            end
                        end

                        work = work*(v1')

                        if n > k
                            #work = work + c2*(v2')
                            work .+= c2*(v2')
                        end

                        if trans == 'C'
                            work = work*(t')
                        else
                            work = work*t
                        end

                        #c2 = c2 - w*v2
                        if n > k
                            c2 .-= work*v2
                        end

                        work = work*v1
                        
                        #c1 = c1 - w

                        for j in 1:k
                            for i in 1:m
                                c[i,j] = c[i,j] - work[i,j]
                            end
                        end

                    end
                end
            else # direct = B
                if side == 'L' || side == 'l'
                    v1 = @view v[:, 1:m-k]
                    v2 = @view v[:, m-k+1:m]
                    c1 = @view c[1:m-k,:]
                    c2 = @view c[m-k+1:m,:]

                    #w = c2^h
                    for i in 1:k
                        for j in 1:n
                            work[j,i] = conj(c2[i,j])
                        end
                    end

                    work = work * (v2')
                    
                    if m > k
                        #work = work + (c1')*(v1')
                        work .+= (c1')*(v1')
                    end

                    if trans == 'N'
                        work = work*(t')
                    else
                        work = work*t
                    end

                    #c1 = c1 - v1^h * w^h
                    if m > k
                        c1 .-= (v1')*(work')
                    end

                    work = work*v2
                    
                    #c2 = c2 - w^h
                    for j in 1:k
                        for i in 1:n
                            c[m-k+j,i] = c[m-k+j,i] - conj(work[i,j])
                        end
                    end

                else 
                    if side == 'R'
                        v1 = @view v[:, 1:n-k]
                        v2 = @view v[:, n-k+1:n]
                        c1 = @view c[:, 1:n-k]
                        c2 = @view c[:,n-k+1:n]

                        #w = c2
                        for i in 1:m
                            for j in 1:k
                                work[i,j] = c2[i,j]
                            end
                        end

                        work = work * (v2')

                        if n > k
                            #work = work + c1*(v1')
                            work .+= c1*(v1')
                        end

                        if trans == 'C'
                            work = work*(t')
                        else
                            work = work*t
                        end

                        #c1 = c1 - w*v1
                        if n > k
                            c1 .-= work*v1
                        end

                        work = work*v2

                        #c2 = c2 - w
                        for j in 1:k
                            for i in 1:m
                                c[i,n-k+j] = c[i,n-k+j] - work[i,j]
                            end
                        end
                    end
                end
            end
        end
    end

    return
end 