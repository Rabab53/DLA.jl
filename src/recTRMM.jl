export trmm!

# Using own implementation for both GEMM and TRMM
# Baseed on out of place implementation to tackle dependencies

# start: start of block A
# end_index: end of block A
# unit_index: whether A is unit triangular
# upper_index: whether A is an upper triaingular matrix


# to copy the results from C back to B: internal OOP implementation
@kernel function lmem_copy_kernel!(output, @Const(input), 
                                                    ::Val{BANK} = Val(1),) where BANK
    I, J = @index(Global, NTuple)
    i, j = @index(Local, NTuple)

    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (N + BANK, M)

    @inbounds tile[i, j] = input[I, J]

    @synchronize

    @inbounds output[I, J] = tile[i, j]
end


@kernel function gemm_trmm_kernel!(A,B, C,
                            ::Val{BANK} = Val(1)) where BANK
    
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]
    BLOCK_ROWS = @uniform @groupsize()[2]

    #allocating shared memory for the sub matrix product calculation
    #BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    C_sub = @private eltype(B) 1
    @inbounds C_sub[1] = -zero(eltype(B))

    @uniform N = size(A, 1)
    @uniform R = size(A, 2)
    @uniform M = size(B, 2)


    #the number of tiles required will be dependent on the inner dimensions
    @uniform NUM_TILES = div(R + TILE_DIM - 1, TILE_DIM)

    #loop over all tiles needed for the calculation
    for t in 0:(NUM_TILES-1)
        # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
        I = (gi-1) * TILE_DIM + i
        J = (gj-1) * TILE_DIM + j

        # load inputs into tiles, with bounds checking for non-square matrices
        if I <= N && t*TILE_DIM + j <= R
            @inbounds tile1[i, j] = A[I, t*TILE_DIM + j]
        else
            @inbounds tile1[i, j] = 0.0
        end
        if t*TILE_DIM + i <= R && J <= M
            @inbounds tile2[i, j] = B[t*TILE_DIM + i, J]
        else
            @inbounds tile2[i, j] = 0.0
        end

        # wait for all tiles to be loaded
        @synchronize

        # get global values again
        I = (gi-1) * TILE_DIM + i
        J = (gj-1) * TILE_DIM + j

        # calculate value of spot in output, use temporary value to allow for vectorization
        out = zero(eltype(B))
        @simd for k in 1:TILE_DIM
            @inbounds out += tile1[i, k] * tile2[k, j]
        end
        C_sub[1] += out

        @synchronize
    end

    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds C[I, J] += C_sub[1]
    end
    @synchronize
end


# mimics the gemm kernel, can be changed to a specialized trmm kernel
@kernel function createTRMMBlockKernel!(A,B,C,
                            ::Val{BANK} = Val(1)) where BANK
    
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]
    BLOCK_ROWS = @uniform @groupsize()[2]

    #allocating shared memory for the sub matrix product calculation
    #BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(C) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(C) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    C_sub = @private eltype(C) 1
    @inbounds C_sub[1] = -zero(eltype(C))

    @uniform N = size(A, 1)
    @uniform R = size(A, 2)
    @uniform M = size(B, 2)


    #the number of tiles required will be dependent on the inner dimensions
    @uniform NUM_TILES = div(R + TILE_DIM - 1, TILE_DIM)

    #loop over all tiles needed for the calculation
    for t in 0:(NUM_TILES-1)
        # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
        I = (gi-1) * TILE_DIM + i
        J = (gj-1) * TILE_DIM + j

        # load inputs into tiles, with bounds checking for non-square matrices
        if I <= N && t*TILE_DIM + j <= R
            @inbounds tile1[i, j] = A[I, t*TILE_DIM + j]
        else
            @inbounds tile1[i, j] = 0.0
        end
        if t*TILE_DIM + i <= R && J <= M
            @inbounds tile2[i, j] = B[t*TILE_DIM + i, J]
        else
            @inbounds tile2[i, j] = 0.0
        end

        # wait for all tiles to be loaded
        @synchronize

        # get global values again (because of synchronize?)
        I = (gi-1) * TILE_DIM + i
        J = (gj-1) * TILE_DIM + j

        # calculate value of spot in output, use temporary value to allow for vectorization
        out = zero(eltype(C))
        @simd for k in 1:TILE_DIM
            @inbounds out += tile1[i, k] * tile2[k, j]
        end
        C_sub[1] += out

        @synchronize
    end

    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds C[I, J] = C_sub[1]
    end
    @synchronize
end


# @kernel function createTRMMBlockKernel2!(A, B, start_index, end_index, )

function trmm_recursive!(Afull, Bfull, Cfull, start_index, end_index, limit)
    size_tile = end_index - start_index + 1

    # if the matrix is small enough, call the computation kernel directly for the block
    if size_tile <= limit
        # set the kernel arguments
        nthreads = 16
        lWorkSize = (nthreads, nthreads)
        A = @view(Afull[start_index:end_index, start_index: end_index])
        B = @view(Bfull[start_index:end_index, 1:end])
        C = @view(Cfull[start_index:end_index, 1:end])
        

        backend = get_backend(A)
        padded_c = (size(B,1)+nthreads[1], size(B,2)+nthreads[1])
        createTRMMBlockKernel!(backend, lWorkSize)(A, B, C; ndrange = padded_c) 
    
        
    
    else
        # split at the next multiple of the TileSize
        split = div(size_tile, 2)

        # considering the lower triangular case first
        trmm_recursive!(Afull, Bfull, Cfull, start_index+split, end_index, limit)        
        gemm!(Afull, Bfull, Cfull, start_index+split, end_index, start_index, start_index+split - 1, start_index, start_index + split - 1, end_index)
        trmm_recursive!(Afull, Bfull, Cfull, start_index, start_index+split-1, limit)

    end
end




# holder wrapper for the kernel

function trmm!(A, B)
    if size(A)[1] != size(A)[2]
        error("Dimension mismatch: Matrix A must be triangular!")
    end

    if size(A)[2] != size(B)[1]
        error("Matrix A and B not compatible for matrix product!")
    end
    limit = 1024
    nthreads = (16, 16)
    C = similar(B)

    trmm_recursive!(A, B, C, 1, size(A)[1], limit)

    backend = get_backend(A)
    lmem_copy_kernel!(backend, nthreads)(B, C; ndrange = size(C))


end


function gemm!(Afull, Bfull, Cfull, ll_startR, ll_endR, ll_startC, ll_endC, b_upper_start, b_upper_end, end_index; n_threads = (16, 16))

    A = @view(Afull[ll_startR:ll_endR, ll_startC:ll_endC])
    B = @view(Bfull[b_upper_start:b_upper_end, 1:end])
    C = @view(Cfull[b_upper_end+1:end_index, 1:end])

    
    backend = get_backend(A)
    gemm_trmm_kernel!(backend, n_threads)(A, B, C; ndrange = size(C))
    
end

function createBlockTrmm!(A, B, C; n_threads = (16, 16))

    backend = get_backend(A)
    gemm_trmm_kernel!(backend, n_threads)(A, B, C; ndrange = size(C))
    
    

end