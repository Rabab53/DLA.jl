for elty in (Float64, Float32, ComplexF64, ComplexF32)
    @eval begin
        function coreblas_gbtype2cb!(
            uplo, 
            n, 
            nb,
            A::AbstractMatrix{$elty}, 
            VQ::Vector{$elty}, 
            TAUQ::Vector{$elty},
            VP::Vector{$elty}, 
            TAUP::Vector{$elty},
            st,
            ed,
            sweep,
            Vblksiz,
            wantz,
            WORK::Vector{$elty})

            lda = size(A, 1)
            ctmp = Ref{$elty}()
            i = 0
            blkid, vpos, taupos, tpos = 0, 0, 0, 0

            LDX = lda-1
            J1  = ed+1
            J2  = min(ed+nb,n-1)
            lem = ed-st+1
            len = J2-J1+1

            if uplo == CoreBlasUpper
                # /* ========================
                # *       UPPER CASE
                # * ========================*/
                if len > 0
                    if wantz == 0
                        vpos   = ((sweep+1)%2)*n + st
                        taupos = ((sweep+1)%2)*n + st
                    else
                        vpos, taupos, tpos, blkid = findVTpos(
                            n, nb, Vblksiz, sweep, st)
                    end
                    vpos += 1
                    taupos += 1

                    # // Apply remaining Left commming from type1/3_upper 
                    ctmp = conj(TAUQ[taupos])
                    # LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'L',
                    #         lem, len, VQ(vpos), ctmp, AU(st, J1), LDX, WORK);
                    LAPACK.larf!('L', lem, len, pointer(VQ, vpos), 
                                ctmp[], @AU_ptr(st, J1), LDX, WORK)

                end

                if len > 1 
                    if wantz == 0
                        vpos   = ((sweep+1)%2)*n + J1
                        taupos = ((sweep+1)%2)*n + J1
                    else 
                        vpos, taupos, tpos, blkid = findVTpos(n,nb,Vblksiz,sweep,J1)
                    end
                    vpos += 1
                    taupos += 1

                    # // Remove the top row of the created bulge 
                    VP[vpos] = 1.;
                    for i=1:len-1
                        VP[vpos+i] = conj(@AU(st, J1+i))
                        @AU_set(st, J1+i, 0.)
                    end
                    # // Eliminate the row  at st 
                    ctmp[] = conj(@AU(st, J1));
                    # LAPACKE_zlarfg_work(len, &ctmp, VP(vpos+1), 1, TAUP(taupos) );
                    LAPACK.larfg!(len, ctmp, pointer(VP, vpos+1), Ref(TAUP, taupos))
                    @AU_set(st, J1, ctmp[])

                    # /*
                    # * Apply Right on A(J1:J2,st+1:ed)
                    # * We decrease len because we start at row st+1 instead of st.
                    # * row st is the row that has been revomved;
                    # */
                    lem = lem-1
                    ctmp[] = TAUP[taupos]
                    # LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'R',
                    #             lem, len, VP(vpos), ctmp, AU(st+1, J1), LDX, WORK);
                    LAPACK.larf!('R', lem, len, pointer(VP, vpos),
                                ctmp[], @AU_ptr(st+1, J1), LDX, WORK)
                end

            else
                # /* ========================
                # *       LOWER CASE
                # * ========================*/
                if len > 0
                    if wantz == 0
                        vpos   = ((sweep+1)%2)*n + st
                        taupos = ((sweep+1)%2)*n + st
                    else
                        vpos, taupos, tpos, blkid = findVTpos(n, nb, Vblksiz, sweep, st)
                    end
                    vpos += 1
                    taupos += 1

                    # // Apply remaining Right commming from type1/3_lower 
                    ctmp[] = TAUP[taupos];
                    # LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'R',
                    #                 len, lem, VP(vpos), ctmp, AL(J1, st), LDX, WORK);
                    LAPACK.larf!('R', len, lem, pointer(VP, vpos),
                                ctmp[], @AL_ptr(J1, st), LDX, WORK)
                end
                
                if len > 1
                    if wantz == 0
                        vpos   = ((sweep+1)%2)*n + J1
                        taupos = ((sweep+1)%2)*n + J1
                    else
                        vpos, taupos, tpos, blkid = findVTpos(n,nb,Vblksiz,sweep,J1)
                    end
                    vpos += 1
                    taupos += 1

                    # // Remove the first column of the created bulge
                    VQ[vpos] = 1.
                    # memcpy(VQ(vpos+1), AL(J1+1, st), (len-1)*sizeof(coreblas_complex64_t));
                    # memset(AL(J1+1, st), 0, (len-1)*sizeof(coreblas_complex64_t));
                    VQ[vpos+1:vpos+len-1] .= @ALv_(J1+1, st, len-1)
                    @ALv_set(J1+1, st, 0, len-1)
                    # // Eliminate the col  at st 
                    # LAPACKE_zlarfg_work(len, AL(J1, st), VQ(vpos+1), 1, TAUQ(taupos) );
                    LAPACK.larfg!(len, @AL_ptr(J1, st), pointer(VQ, vpos+1), Ref(TAUQ, taupos))

                    # /*
                    # * Apply left on A(J1:J2,st+1:ed)
                    # * We decrease len because we start at col st+1 instead of st.
                    # * col st is the col that has been revomved;
                    # */
                    lem = lem-1
                    ctmp[] = conj(TAUQ[taupos])
                    
                    # LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'L',
                    #             len, lem, VQ(vpos), ctmp, AL(J1, st+1), LDX, WORK);
                    LAPACK.larf!('L', len, lem, pointer(VQ, vpos),
                                ctmp[], @AL_ptr(J1, st+1), LDX, WORK)  
                end
                
            end
            # // end of uplo case
            return

        end

        # WORK included
        function coreblas_gbtype2cb!(
            uplo,
            n, 
            nb, 
            A::AbstractMatrix{$elty}, 
            VQ::Vector{$elty},  
            TAUQ::Vector{$elty}, 
            VP::Vector{$elty},  
            TAUP::Vector{$elty},
            st,
            ed,
            sweep,
            Vblksiz,
            wantz)
        
            WORK = Vector{$elty}(undef, nb)
        
            coreblas_gbtype2cb!(uplo, n, nb, A, VQ, TAUQ, VP, TAUP, st, ed, sweep, Vblksiz, wantz, WORK)
        end

    end
end