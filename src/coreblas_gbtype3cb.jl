for elty in (Float64, Float32, ComplexF64, ComplexF32)
    @eval begin
        function coreblas_gbtype3cb!(
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
            i, len, LDX, lenj = 0, 0, 0, 0
            blkid, vpos, taupos, tpos = 0, 0, 0, 0

            if wantz == 0
                vpos   = ((sweep+1)%2)*n + st;
                taupos = ((sweep+1)%2)*n + st;
            else 
                vpos, taupos, tpos, blkid = findVTpos(n, nb, Vblksiz, sweep, st);
            end
            vpos += 1
            taupos += 1

            LDX = lda-1;
            len = ed-st+1;
            if uplo == CoreBlasUpper
                # /* ========================
                # *       UPPER CASE
                # * ========================*/
                # // Apply right on A(st:ed,st:ed) 
                ctmp[] = TAUP[taupos]
                # LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'R',
                #                 len, len, VP(vpos), ctmp, AU(st, st), LDX, WORK);
                LAPACK.larf!('R', len, len, pointer(VP, vpos),
                            ctmp[], @AU_ptr(st, st), LDX, WORK)

                # // Eliminate the created col at st 
                VQ[vpos] = 1.
                # memcpy( VQ(vpos+1), AU(st+1, st), (len-1)*sizeof(coreblas_complex64_t) );
                # memset( AU(st+1, st), 0, (len-1)*sizeof(coreblas_complex64_t) );
                VQ[vpos+1:vpos+len-1] .= @AUv_(st+1, st, len-1)
                @AUv_set(st+1, st, 0, len-1)
                # LAPACKE_zlarfg_work(len, AU(st, st), VQ(vpos+1), 1, TAUQ(taupos) );
                LAPACK.larfg!(len, @AU_ptr(st, st), pointer(VQ, vpos+1), Ref(TAUQ, taupos))

                lenj = len-1
                ctmp[] = conj(TAUQ[taupos])
                
                # LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'L',
                #                 len, lenj, VQ(vpos), ctmp, AU(st, st+1), LDX, WORK);
                LAPACK.larf!('L', len, lenj, pointer(VQ, vpos),
                            ctmp[], @AU_ptr(st, st+1), LDX, WORK)

            else
                # /* ========================
                # *       LOWER CASE
                # * ========================*/
                # // Apply left on A(st:ed,st:ed) 
                ctmp[] = conj(TAUQ[taupos])
                # LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'L',
                #             len, len, VQ(vpos), ctmp, AL(st, st), LDX, WORK);
                LAPACK.larf!('L', len, len, pointer(VQ, vpos),
                            ctmp[], @AL_ptr(st, st), LDX, WORK)

                # // Eliminate the created row at st
                VP[vpos] = 1.
                for i in 1:len-1
                    VP[vpos+i]     = conj(@AL(st, st+i))
                    @AL_set(st, st+i, 0.)
                end
                ctmp[] = conj(@AL(st, st))

                # LAPACKE_zlarfg_work(len, &ctmp, VP(vpos+1), 1, TAUP(taupos) );
                LAPACK.larfg!(len, ctmp, pointer(VP, vpos+1), Ref(TAUP, taupos))

                @AL_set(st, st, ctmp[])
                lenj = len-1
                ctmp[] = TAUP[taupos]
                # LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'R',
                #                 lenj, len, VP(vpos), ctmp, AL(st+1, st), LDX, WORK);
                LAPACK.larf!('R', lenj, len, pointer(VP, vpos), 
                            ctmp[], @AL_ptr(st+1, st), LDX, WORK)

            end
            # // end of uplo case 
            return;
        end

        # WORK included
        function coreblas_gbtype3cb!(
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
        
            coreblas_gbtype3cb!(uplo, n, nb, A, VQ, TAUQ, VP, TAUP, st, ed, sweep, Vblksiz, wantz, WORK)
        end

    end
end