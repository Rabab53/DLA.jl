function coreblas_zgbtype1cb!(
    uplo, 
    n, 
    nb,
    A::AbstractMatrix, 
    lda,
    VQ::Vector, 
    TAUQ::Vector,
    VP::Vector, 
    TAUP::Vector,
    st,
    ed,
    sweep,
    Vblksiz,
    wantz,
    WORK::Vector)

    # lda = size(A, 1)

    ctmp = 0
    i, len, LDX, lenj = 0, 0, 0, 0
    blkid, vpos, taupos, tpos = 0, 0, 0, 0

    # /* find the pointer to the Vs and Ts as stored by the bulgechasing
    # * note that in case no eigenvector required V and T are stored
    # * on a vector of size n
    # * */
    if wantz == 0
        vpos   = ((sweep+1)%2)*n + st
        taupos = ((sweep+1)%2)*n + st
    else 
        findVTpos(n, nb, Vblksiz, sweep, st,
                  Ref(vpos), Ref(taupos), Ref(tpos), Ref(blkid))
    end

    vpos += 1
    taupos += 1

    LDX = lda-1
    len = ed-st+1

    if uplo == CoreBlasUpper
        # /* ========================
        #  *       UPPER CASE
        #  * ========================*/
        # // Eliminate the row  at st-1 
        
        VP[vpos] = 1.
        for i in 1:len-1
            VP[vpos+i] = conj(A[2*nb-i+1, st+i+1]);
            A[2*nb-i+1, st+i+1] = 0.;
        end
        ctmp = conj(A[2*nb+1, st+1])
        
        # LAPACKE_zlarfg(len, &ctmp, VP(vpos+1), 1, TAUP(taupos) );
        TAUP[taupos] = LAPACK.larfg!(@view(VP[vpos+1:vpos+len-1]));
        A[2*nb, st+1] = ctmp;
        # // Apply right on A(st:ed,st:ed) 
        ctmp = TAUP[taupos];
            
        # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'R',
        #                 len, len, VP(vpos), ctmp, AU(st, st), LDX, WORK);
        LAPACK.larf!('R', @view(VP[vpos:vpos+len-1]), ctmp, @view(A[2*nb+st+1:2*nb+ed+1,st+1:ed+1]), WORK)

        # // Eliminate the created col at st 
        VQ[vpos] = 1.;
        # memcpy( @VQ(vpos+1), AU(st+1, st), (len-1)*sizeof(coreblas_complex64_t) );
        # memset( AU(st+1, st), 0, (len-1)*sizeof(coreblas_complex64_t) );
        VQ[vpos+1:vpos+len-1] .= A[2*nb+1:2*nb+len-1,st+1]
        A[2*nb+2:2*nb+1+len,st+1]= 0.

        # LAPACKE_zlarfg(len, AU(st, st), VQ(vpos+1), 1, TAUQ(taupos) );
        TAUQ[taupos] = LAPACK.larfg!(@view(VQ[vpos+1:vpos+len-1]))

        lenj = len-1;
        ctmp = conj(TAUQ[taupos]);

        # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'L',
        #          len, lenj, VQ(vpos), ctmp, AU(st, st+1), LDX, WORK);
        LAPACK.larf!('L', @view(VP[vpos:vpos+len-1]), ctmp, @view(A[2*nb+st:2*nb+ed,st+2:ed+1]), WORK)

    # else 
    #     # /* ========================
    #     #  *       LOWER CASE
    #     #  * ========================*/
    #     # // Eliminate the col  at st-1

    #     @VQ(vpos)[] = 1.;
    #     # TODO: memcpy/memset
    #     # memcpy( VQ(vpos+1), AL(st+1, st-1), (len-1)*sizeof(coreblas_complex64_t) );
    #     # memset( AL(st+1, st-1), 0, (len-1)*sizeof(coreblas_complex64_t) );

    #     # TODO: zlarfg
    #     # LAPACKE_zlarfg(len, AL(st, st-1), VQ(vpos+1), 1, TAUQ(taupos) );

    #     # // Apply left on A(st:ed,st:ed) 
    #     ctmp = conj(@TAUQ(taupos)[]);

    #     # TODO: zlarfx
    #     # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'L',
    #     #             len, len, VQ(vpos), ctmp, AL(st, st), LDX, WORK);

    #     # // Eliminate the created row at st 
    #     @VP(vpos)[] = 1.;

    #     for i in 1:(len-1)
    #         @show vpos
    #         @VP(vpos+i)[] = conj(@AL(st, st+i)[]);
    #         @AL(st, st+i)[] = 0.;
    #     end
    #     ctmp = conj(@AL(st, st)[]);
        
    #     # TODO: zlarfg
    #     # LAPACKE_zlarfg(len, &ctmp, VP(vpos+1), 1, TAUP(taupos) );

    #     @AL(st, st)[] = ctmp;
    #     lenj = len-1;
    #     ctmp = (@TAUP(taupos)[]);
    #     # TODO: zlarfx
    #     # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'R',
    #     #                 lenj, len, VP(vpos), ctmp, AL(st+1, st), LDX, WORK);

    end # // end of uplo case

    return;
end