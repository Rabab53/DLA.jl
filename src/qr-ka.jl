using CUDA, LinearAlgebra, KernelAbstractions, Test
using KernelAbstractions.Extras: @unroll

### KA kernels for tiled QR and SVD

### input: square input matrix of tiltsize 32,32 and tau array of size 32
### output: input matrix contains R in upper triangular part, 
### and householder reflector information in lower triangular part + in tau array
@kernel function QR_unsafe_kernel_2d!(input, tau) 
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (N + 1, N)
    cache = @localmem eltype(input) (N + 1)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @inbounds tile[i, j] = input[i, j]


    for iter in 1:N-1
        if (i > iter) && (j == iter)
            cache[i] = tile[i, iter]^2
        end
        @synchronize
        if (i == 1) && (j == 1)
            tmp_sum = zero(eltype(input))
            for l in iter+1:N
                tmp_sum += cache[l]
            end
            tmp_sum2 = sqrt(tmp_sum + tile[iter, iter]^2)
            newvalue = tile[iter, iter] + sign(tile[iter, iter]) * tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
        end
        if (j >= iter) && (i >= iter)
            tmp_sum = zero(eltype(input))
            for k = iter+1:N
                tmp_sum += tile[k, iter]  * tile[k, j]
            end
        end
        tileiterj=tile[iter, j]
        tileiiter = tile[i, iter] 
        @synchronize
        if (j >= iter) && (i >= iter)
            corrvalue1 = corrvalue[1]
            tmp_sum = (tmp_sum / corrvalue1+ tileiterj)*tau_iter[1] 
            tileiiter = tileiiter / corrvalue1

            if (j==iter) && (i > iter) 
                tile[i, j] = tileiiter 
            elseif (i>iter)
                tile[i, j] = tile[i, j] - tileiiter* tmp_sum  
            else
                tile[i, j] = tile[i, j] - tmp_sum 
            end
        end
        @synchronize
    end
    @inbounds input[i, j] = tile[i, j]
    @synchronize

end

### input: square input matrix of tiltsize 32,32 containing an R factor and 
### housholder reflectors in lower triangular part,
### square input2 matrix containing the input
### calculates the QR factorization of [input.R;input2] and saves the householder reflectors in
### output:: input2 contains householder vector information and tau contains the tau factors
@kernel function QR_unsafe_kernel2_2d!(input, input2, tau)
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (2N + 1, N)
    cache = @localmem eltype(input) (2N + 1)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @inbounds tile[N+i, j] = input2[i, j]
    @inbounds tile[i, j] = input[i, j]
    
    @synchronize
    for iter in 1:N
        if (j==iter)
            cache[i] = tile[i+N, iter]^2
        end
        @synchronize
        if (i == 1) && (j==1)
            tmp_sum = zero(eltype(input))
            for l in 1:N
                tmp_sum += cache[l]
            end
            tmp_sum2 = sqrt(tmp_sum + tile[iter, iter]^2)
            newvalue = tile[iter, iter] + sign(tile[iter,iter]) *tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
        end
        tileiNiter= tile[i+N, iter]
        tileiterj=tile[iter, j]
        if (j>=iter)
            tmp_sum = zero(eltype(input))
            for l = N+1:2N
                tmp_sum += tile[l, iter] * tile[l, j]
            end
        end
        @synchronize
        taucorr=tau_iter[1] / corrvalue[1]
        corrvalue1 = corrvalue[1]
        if (j >= iter) 
            tmp_sum += corrvalue1 * tileiterj
            if (i==iter)
                tile[i, j] = tile[i,j] - tmp_sum * taucorr
            end
            if (j>iter)
                tile[i+N, j] = tile[i+N, j] - tileiNiter * tmp_sum *taucorr / corrvalue1
            end
        end
        if (j==1)
            tile[i+N, iter] = tileiNiter / corrvalue1
        end
        @synchronize
    end
    
    @inbounds input2[i, j] = tile[N+i, j]
    @inbounds input[i, j] = tile[i, j]
    @synchronize

end

### Min and tau are the output of QR_unsafe_kernel_2d!
### function calculations the application of the householder reflectors on 
### input: the square matrix blocks of size 32,32 lined up in a row in A
### and returns the A multiplied by the Q-factor of the QR factorization calculated by  QR_unsafe_kernel_2d!
@kernel function applyQorQt_unsafe_kernel_2d!(A, @Const(Min), @Const(tau), applyt::Bool)
    g, _ = @index(Group, NTuple)
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]
    K = @uniform @groupsize()[2]
    tile = @localmem eltype(A) (N + 1, N)
    M = @localmem eltype(A) (N + 1, N)
    cache = @localmem eltype(A) (N + 1,K)
    
    @unroll for l in j:K:N
        @inbounds tile[i, l] = A[i, l+(g-1)*N]
        @inbounds M[i, l] = Min[i, l]
    end

    applyrange = applyt ? (1:N-1) : (N-1:-1:1)
    
    @synchronize
    for k in applyrange
        tmp_sum = zero(eltype(A))
        for l in k+j:K:N
            tmp_sum += M[l, k] * tile[l, i]
        end
        cache[i,j]=tmp_sum
        @synchronize
        tmp_sum = tile[k, i] 
        for l in 1:K
            tmp_sum+=cache[i,l]
        end
        tmp_sum = tmp_sum * tau[k]
        for l in k+j:K:N
            tile[l, i] = tile[l, i] - tmp_sum * M[l, k]
        end
        if (j==1)
            tile[k, i] = tile[k, i] - tmp_sum
        end
        @synchronize
    end
    @unroll for l in j:K:N
        @inbounds A[i, l+(g-1)*N]  = tile[i, l]
    end
end

### Min and tau are the output of QR_unsafe_kernel2_2d! (only Min=input2 is used, as input is assumed to previously has been applied already)
### function calculations the application of the householder reflectors in Min on 
### A and B: the tall matrix blocks of size 32,32 their upper part lined up in A, their lower part lined up in B
### and returns the A and B multiplied by the Q-factor of the QR factorization calculated by  QR_unsafe_kernel2_2d!
@kernel function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau), applyt::Bool)
    g, _ = @index(Group, NTuple)
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]
    K = @uniform @groupsize()[2]
    tile = @localmem eltype(A) (2N + 1, N)
    M = @localmem eltype(A) (N + 1, N)
    cache = @localmem eltype(A) (N+1, K)
    
    @unroll for l in j:K:N
        @inbounds tile[i, l] = A[i, l+(g-1)*N]
        @inbounds tile[i+N, l] = B[i, l+(g-1)*N]
        @inbounds M[i, l] = Min[i, l]
    end

    applyrange = applyt ? (1:N) : (N:-1:1)

    @synchronize
    for k in applyrange
        tmp_sum= zero(eltype(A))       
        for j in j:K:N
            tmp_sum += M[j, k] * tile[j+N, i]
        end
        cache[i,j]=tmp_sum
        @synchronize
        tmp_sum = tile[k, i]
        for l in 1:K
            tmp_sum+=cache[i,l]
        end
        tmp_sum = tmp_sum * tau[k]
        if (j==1)
            tile[k, i] = tile[k, i] - tmp_sum
        end
        for l in j:K:N
            tile[l+N, i] = tile[l+N, i] - tmp_sum * M[l, k]
        end
        @synchronize
    end
    @unroll for l in j:K:N
        @inbounds A[i, l+(g-1)*N] = tile[i, l]
        @inbounds B[i, l+(g-1)*N] = tile[i+N, l]
    end
end


##### TEST ######
@testset "QRkernels 2D with elty = $elty" for elty in [ Float32, Float16, Float64] 
    n=32
    backend=KernelAbstractions.get_backend(CUDA.randn(2))
    T=elty
    myrange=(n,n)
    t= KernelAbstractions.zeros(backend, T, n)
    
    @testset "single tile QR" begin
    
        a=rand!(allocate(backend, T,n, n))
        acopy=copy(a)
        
        QR_unsafe_kernel_2d!(backend,myrange)(acopy,t, ndrange=myrange)
        @test Array(acopy) ≈ qr(Array(a)).factors

        b=rand!(allocate(backend, T,n, n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel_2d!(backend,myrange)(bcopy, acopy,t, true, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array(a)).Q'*Array(b)

        bcopy=copy(b)
        applyQorQt_unsafe_kernel_2d!(backend,myrange)(bcopy, acopy,t, false, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array(a)).Q*Array(b)

        bcopy=copy(b)
        applyQorQt_unsafe_kernel_2d!(backend,myrange)(bcopy', acopy,t, true, ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array(a)).Q

        bcopy=copy(b)
        applyQorQt_unsafe_kernel_2d!(backend,myrange)(bcopy', acopy,t, false, ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array(a)).Q'

    end

    @testset "double tile QR" begin 
        a=rand!(allocate(backend, T,2n, n))
        acopy=copy(a)
        QR_unsafe_kernel2_2d!(backend,myrange)(view(acopy,1:n,1:n),view(acopy,n+1:2n,1:n), t , ndrange=myrange)
        tril(view(acopy,1:n,1:n),-1) ≈ tril(view(a,1:n,1:n),-1)
        @test  [triu(Array(view(acopy,1:n,1:n)));Array(view(acopy,n+1:2n,1:n))] ≈ qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).factors
        
        b=rand!(allocate(backend, T,2n, n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2_2d!(backend,myrange)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,true, ndrange=myrange )
        @test Array(bcopy) ≈ qr(Array([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))])).Q'*Array(b)

        b=rand!(allocate(backend, T,2n, n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2_2d!(backend,myrange)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,false,ndrange=myrange )
        @test Array(bcopy) ≈ qr([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))]).Q*Array(b)

        b=rand!(allocate(backend, T,n, 2n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2_2d!(backend,myrange)(view(bcopy,1:n,1:n)',view(bcopy,1:n, n+1:2n)',view(acopy,n+1:2n,1:n) , t,true,ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).Q

        b=rand!(allocate(backend, T,n, 2n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2_2d!(backend,myrange)(view(bcopy,1:n,1:n)',view(bcopy,1:n, n+1:2n)',view(acopy,n+1:2n,1:n) , t,false,ndrange=myrange )
        @test Array(bcopy) ≈ Array(b)*qr(Array([triu(view(a,1:n,1:n));view(a,n+1:2n,1:n)])).Q'
   
    end

    @testset "block QR" begin 
        myredrange=(n,Int(n/2))
        myextrange=(5n,Int(n/2))

        a=rand!(allocate(backend, T,n, n))
        acopy=copy(a)
        t= KernelAbstractions.zeros(backend, T, n)
        QR_unsafe_kernel_2d!(backend,myrange)(acopy,t, ndrange=myrange)

        b=rand!(allocate(backend, T,n, 5n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel_2d!(backend,myredrange)(bcopy, acopy,t, true, ndrange=myextrange )
        @test Array(bcopy) ≈ qr(Array(a)).Q'*Array(b)

        a=rand!(allocate(backend, T,2n, n))
        acopy=copy(a)
        QR_unsafe_kernel2_2d!(backend,myrange)(view(acopy,1:n,1:n),view(acopy,n+1:2n,1:n), t , ndrange=myrange)
   
        b=rand!(allocate(backend, T,2n, 5n))
        bcopy=copy(b)
        applyQorQt_unsafe_kernel2_2d!(backend,myredrange)(view(bcopy,1:n,1:n),view(bcopy,n+1:2n,1:n),view(acopy,n+1:2n,1:n) , t,true,ndrange=myextrange)
        @test Array(bcopy) ≈ qr(Array([triu(Array(view(a,1:n,1:n)));Array(view(a,n+1:2n,1:n))])).Q'*Array(b)

    end

end
