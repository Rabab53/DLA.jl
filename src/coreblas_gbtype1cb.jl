## TYPE 1-BAND-bidiag Lower/Upper columnwise-Householder

for elty in (Float64, Float32, ComplexF64, ComplexF32)
    @eval begin
        function coreblas_gbtype1cb!(
            uplo, 
            n, 
            nb,
            A::AbstractMatrix{$elty}, 
            # lda,
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
            # lda = max(1, stride(A,2))

            ctmp = Ref{Float64}()
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
                    VP[vpos+i] = conj(A[2*nb-i, st+1+i]);
                    A[2*nb-i, st+1+i] = 0.;
                end
                ctmp[] = conj(A[2*nb, st+1])

                # LAPACKE_zlarfg(len, &ctmp, VP(vpos+1), 1, TAUP(taupos) );
                LAPACK.larfg!(len, ctmp, pointer(VP, vpos+1), Ref(TAUP, taupos))

                A[2*nb, st+1] = ctmp[];
                # // Apply right on A(st:ed,st:ed) 
                ctmp[] = TAUP[taupos];
                    
                # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'R',
                #                 len, len, VP(vpos), ctmp, AU(st, st), LDX, WORK);
                LAPACK.larf!('R', len, len, pointer(VP, vpos),
                            ctmp[], pointer(A, 2*nb+1+st*lda), LDX, WORK)

                # Eliminate the created col at st 
                VQ[vpos] = 1.;
                VQ[vpos+1:vpos+len-1] .= A[2*nb+2:2*nb+len,st+1]
                A[2*nb+2:2*nb+len,st+1] .= 0

                # LAPACKE_zlarfg(len, AU(st, st), VQ(vpos+1), 1, TAUQ(taupos) );
                ctmp[] = A[2*nb+1,st+1]
                LAPACK.larfg!(len, ctmp, pointer(VQ, vpos+1), Ref(TAUQ, taupos))
                A[2*nb+1,st+1] = ctmp[]

                lenj = len-1;
                ctmp[] = conj(TAUQ[taupos]);

                # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'L',
                #          len, lenj, VQ(vpos), ctmp, AU(st, st+1), LDX, WORK);
                LAPACK.larf!('L', len, lenj, pointer(VQ, vpos),
                    ctmp[], pointer(A, 2*nb+lda*(st+1)), LDX, WORK)

            else 
                # /* ========================
                #  *       LOWER CASE
                #  * ========================*/
                # // Eliminate the col  at st-1

                VQ[vpos] = 1.
                # memcpy( VQ(vpos+1), AL(st+1, st-1), (len-1)*sizeof(coreblas_complex64_t) );
                # memset( AL(st+1, st-1), 0, (len-1)*sizeof(coreblas_complex64_t) );
                VQ[vpos+1:vpos+len-1] .= A[nb+3:nb+len+1,st]
                A[nb+3:nb+len+1,st] .= 0

                # LAPACKE_zlarfg(len, AL(st, st-1), VQ(vpos+1), 1, TAUQ(taupos) )
                ctmp[] = A[nb+2,st]
                LAPACK.larfg!(len, ctmp, pointer(VQ, vpos+1), Ref(TAUQ, taupos))
                A[nb+2,st] = ctmp[]

                # // Apply left on A(st:ed,st:ed) 
                ctmp[] = conj(TAUQ[taupos]);

                # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'L',
                #             len, len, VQ(vpos), ctmp, AL(st, st), LDX, WORK);
                LAPACK.larf!('L', len, len, pointer(VQ, vpos),
                            ctmp[], pointer(A, nb+1+st*lda), LDX, WORK)

                # // Eliminate the created row at st 
                VP[vpos] = 1.;

                for i in 1:(len-1)
                    VP[vpos+i] = A[nb+1-i,st+1+i]
                    A[nb+1-i,st+1+i] = 0.
                end
                ctmp[] = conj(A[nb+1,st+1])
                
                # LAPACKE_zlarfg(len, &ctmp, VP(vpos+1), 1, TAUP(taupos) );
                LAPACK.larfg!(len, ctmp, pointer(VP, vpos+1), Ref(TAUP, taupos))
                A[nb+1,st+1] = ctmp[]

                lenj = len-1;
                ctmp = TAUP[taupos]
                # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'R',
                #                 lenj, len, VP(vpos), ctmp, AL(st+1, st), LDX, WORK);
                LAPACK.larf!('R', lenj, len, pointer(VP, vpos),
                            ctmp[], pointer(A, nb+2+st*lda), LDX, WORK)

            end 
            
            # // end of uplo case
            return;
        end

        # WORK included
        function coreblas_gbtype1cb!(
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
        
            coreblas_gbtype1cb!(uplo, n, nb, A, VQ, TAUQ, VP, TAUP, st, ed, sweep, Vblksiz, wantz, WORK)
        end

    end
end
