# macro AL(m_, n_)
#     return :(A + nb + lda * ($n_) + (($m_)-($n_)))
# end

# macro AU(m_, n_)
#     return :(A + nb + lda * ($n_) + (($m_)-($n_)+nb))
# end

# macro VQ(m)
#     return :(VQ + (m))
# end

# macro VP(m)
#     return :(VP + (m))
# end

# macro TAUQ(m)
#     return :(TAUQ + (m))
# end

# macro TAUP(m)
#     return :(TAUP + (m))
# end 

# yes the repeated reference and dereferencing is a bit silly right now
# will deal with it later after ensuring correctness, since this mirrors the original more closely

macro AL(m_, n_)
    return esc(:(Ref(A[], nb + lda * ($n_) + (($m_)-($n_)) +1)))
end

macro AU(m_, n_)
    quote
        Ref(A[], nb + lda * $(esc(n_)) + ($(esc(m_))-$(esc(n_))+nb) +1)
    end
end

macro VQ(m)
    # quote
    #     Ref(VQ[], $(esc(m)) +1)
    # end
    return esc(:(Ref(VQ[], ($m) +1)))
end

macro VP(m)
    # quote 
    #     Ref(VP[], $(esc(m)) +1)
    # end
    return esc(:(Ref(VP[], ($m) +1)))
end

macro TAUQ(m)
    # quote 
    #     Ref(TAUQ[], $(esc(m)) +1)
    # end
    return esc(:(Ref(TAUQ[], ($m) +1)))
end

macro TAUP(m)
    # quote
    #     Ref(TAUP[], $(esc(m)) +1)
    # end
    return esc(:(Ref(TAUP[], ($m) +1)))
end 


function coreblas_zgbtype1cb!(
    uplo::coreblas_enum_t, 
    n::Int, 
    nb::Int,
    A::Base.RefValue{Matrix{coreblas_complex64_t}}, 
    lda::Int,
    VQ::Base.RefValue{Vector{coreblas_complex64_t}}, 
    TAUQ::Base.RefValue{Vector{coreblas_complex64_t}},
    VP::Base.RefValue{Vector{coreblas_complex64_t}}, 
    TAUP::Base.RefValue{Vector{coreblas_complex64_t}},
    st::Int,
    ed::Int,
    sweep::Int,
    Vblksiz::Int,
    wantz::Int,
    WORK::Base.RefValue{Vector{coreblas_complex64_t}})

    ctmp::coreblas_complex64_t = 0
    i, len, LDX, lenj ::Int = 0, 0, 0, 0
    blkid, vpos, taupos, tpos ::Int = 0, 0, 0, 0

    # /* find the pointer to the Vs and Ts as stored by the bulgechasing
    # * note that in case no eigenvector required V and T are stored
    # * on a vector of size n
    # * */
    if wantz == 0
        vpos   = ((sweep+1)%2)*n + st;
        taupos = ((sweep+1)%2)*n + st;
    else 
        findVTpos(n, nb, Vblksiz, sweep, st,
                  Ref(vpos), Ref(taupos), Ref(tpos), Ref(blkid));
    end

    LDX = lda-1;
    len = ed-st+1;

    if uplo == CoreBlasUpper
        # /* ========================
        #  *       UPPER CASE
        #  * ========================*/
        # // Eliminate the row  at st-1 
        
        @VP(vpos)[] = 1.;
        for i in 1:(n-1)
            @VP(vpos+i)[] = conj(@AU(st-1, st+i))[];
            @AU(st-1, st+i)[] = 0.;
        end
        ctmp = conj(@AU(st-1, st)[]);
        
        # TODO: zlarfg ????
        LAPACK.larfg!(@VP(vpos+1));
        @AU(st-1, st)[] = ctmp;
        # // Apply right on A(st:ed,st:ed) 
        ctmp = @TAUP(taupos)[];
            
        # TODO: zlarfx ????
        # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'R',
        #                 len, len, VP(vpos), ctmp, AU(st, st), LDX, WORK);

        # // Eliminate the created col at st 
        @VQ(vpos)[] = 1.;
        # memcpy( @VQ(vpos+1), AU(st+1, st), (len-1)*sizeof(coreblas_complex64_t) );
        # memset( AU(st+1, st), 0, (len-1)*sizeof(coreblas_complex64_t) );
        for i=0:(len-2)
            @VQ(vpos+1+i)[] = @AU(st+1, st+i)[]
            @AU(st+1, st+i)[] = 0
        end

        # TODO: zlarfg???
        # LAPACKE_zlarfg64_(len, AU(st, st), VQ(vpos+1), 1, TAUQ(taupos) );
        
        lenj = len-1;
        ctmp = conj(@TAUQ(taupos)[]);

        #TODO: zlarfx???
        # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'L',
        #          len, lenj, VQ(vpos), ctmp, AU(st, st+1), LDX, WORK);

    else 
        # /* ========================
        #  *       LOWER CASE
        #  * ========================*/
        # // Eliminate the col  at st-1

        @VQ(vpos)[] = 1.;
        # TODO: memcpy/memset
        # memcpy( VQ(vpos+1), AL(st+1, st-1), (len-1)*sizeof(coreblas_complex64_t) );
        # memset( AL(st+1, st-1), 0, (len-1)*sizeof(coreblas_complex64_t) );

        # TODO: zlarfg
        # LAPACKE_zlarfg(len, AL(st, st-1), VQ(vpos+1), 1, TAUQ(taupos) );

        # // Apply left on A(st:ed,st:ed) 
        ctmp = conj(@TAUQ(taupos)[]);

        # TODO: zlarfx
        # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'L',
        #             len, len, VQ(vpos), ctmp, AL(st, st), LDX, WORK);

        # // Eliminate the created row at st 
        @VP(vpos)[] = 1.;

        for i in 1:(len-1)
            @show vpos
            @VP(vpos+i)[] = conj(@AL(st, st+i)[]);
            @AL(st, st+i)[] = 0.;
        end
        ctmp = conj(@AL(st, st)[]);
        
        # TODO: zlarfg
        # LAPACKE_zlarfg(len, &ctmp, VP(vpos+1), 1, TAUP(taupos) );

        @AL(st, st)[] = ctmp;
        lenj = len-1;
        ctmp = (@TAUP(taupos)[]);
        # TODO: zlarfx
        # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'R',
        #                 lenj, len, VP(vpos), ctmp, AL(st+1, st), LDX, WORK);

    end # // end of uplo case

    return;
end