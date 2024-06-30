## TYPE 1-BAND-bidiag Lower/Upper columnwise-Householder

macro AL(m_, n_)
    return esc(quote
        A[nb+($m_)-($n_)+1,($n_)+1]
    end)
end

macro AU(m_, n_)
    return esc(quote
        A[2*nb+($m_)-($n_)+1,($n_)+1]
    end)
end

macro AL_ptr(m_, n_)
    return esc(quote
        pointer(A, nb+($m_)-($n_)+1+(($n_))*lda)
    end)
end

macro AU_ptr(m_, n_)
    return esc(quote
        pointer(A, 2*nb+($m_)-($n_)+1+($n_)*lda)
    end)
end

macro AL_ref(m_, n_)
    return esc(quote
        Ref(A, nb+($m_)-($n_)+1+(($n_))*lda)
    end)
end

macro AU_ref(m_, n_)
    return esc(quote
        Ref(A, 2*nb+($m_)-($n_)+1+($n_)*lda)
    end)
end

macro ALv_(m_, n_, len_)
    return esc(quote
        A[nb+($m_)-($n_)+1:nb+($m_)-($n_)+($len_),($n_)+1]
    end)
end

macro AUv_(m_, n_, len_)
    return esc(quote
        A[2*nb+($m_)-($n_)+1:2*nb+($m_)-($n_)+($len_),($n_)+1]
    end)
end

macro AL_set(m_, n_, v_)
    return esc(quote
        A[nb+($m_)-($n_)+1,($n_)+1] = ($v_)
    end)
end

macro AU_set(m_, n_, v_)
    return esc(quote
        A[2*nb+($m_)-($n_)+1,($n_)+1] = ($v_)
    end)
end

macro ALv_set(m_, n_, v_, len_)
    return esc(quote
        A[nb+($m_)-($n_)+1:nb+($m_)-($n_)+($len_),($n_)+1] .= ($v_)
    end)
end

macro AUv_set(m_, n_, v_, len_)
    return esc(quote
        A[2*nb+($m_)-($n_)+1:2*nb+($m_)-($n_)+($len_),($n_)+1] .= ($v_)
    end)
end

for elty in (Float64, Float32, ComplexF64, ComplexF32)
    @eval begin
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
            wantz,
            WORK::Vector{$elty})

            # @inline function AL(m_, n_)
            #     return A[nb+m_-n_+1,n_+1]
            # end

            # @inline function AL(m_, n_, len)
            #     return A[nb+m_-n_+1:nb+m_-n_+len,n_+1]
            # end

            # @inline function AU(m_, n_)
            #     return A[2*nb+m_-n_+1,n_+1]
            # end

            # @inline function AU(m_, n_, len)
            #     return A[2*nb+m_-n_+1:2*nb+m_-n_+len,n_+1]
            # end

            lda = size(A, 1)
            # lda = max(1, stride(A,2))

            ctmp = Ref{$elty}()
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
                vpos, taupos, tpos, blkid = findVTpos(
                    n, nb, Vblksiz, sweep, st)
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
                    # VP[vpos+i] = conj(A[2*nb-i, st+1+i]);
                    VP[vpos+i] = conj(@AU(st-1, st+i))
                    # A[2*nb-i, st+1+i] = 0.;
                    @AU_set(st-1, st+i, 0.)
                end
                # ctmp[] = conj(A[2*nb, st+1])
                ctmp[] = conj(@AU(st-1, st))

                # LAPACKE_zlarfg(len, &ctmp, VP(vpos+1), 1, TAUP(taupos) );
                LAPACK.larfg!(len, ctmp, pointer(VP, vpos+1), Ref(TAUP, taupos))

                # A[2*nb, st+1] = ctmp[];
                @AU_set(st-1, st, ctmp[]display("max normalized error: $(maximum(d))")
                display("    acceptable error: $(err)"))
                # // Apply right on A(st:ed,st:ed) 
                ctmp[] = TAUP[taupos];
                    
                # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'R',
                #                 len, len, VP(vpos), ctmp, AU(st, st), LDX, WORK);
                # LAPACK.larf!('R', len, len, pointer(VP, vpos),
                #             ctmp[], pointer(A, 2*nb+1+st*lda), LDX, WORK)
                LAPACK.larf!('R', len, len, pointer(VP, vpos),
                            ctmp[], @AU_ptr(st, st), LDX, WORK)

                # Eliminate the created col at st 
                VQ[vpos] = 1.;
                # VQ[vpos+1:vpos+len-1] .= A[2*nb+2:2*nb+len,st+1]
                # A[2*nb+2:2*nb+len,st+1] .= 0
                VQ[vpos+1:vpos+len-1] .= @AUv_(st+1, st, len-1)
                @AUv_set(st+1, st, 0, len-1)

                # LAPACKE_zlarfg(len, AU(st, st), VQ(vpos+1), 1, TAUQ(taupos) );
                # ctmp[] = A[2*nb+1,st+1]
                # LAPACK.larfg!(len, ctmp, pointer(VQ, vpos+1), Ref(TAUQ, taupos))
                # A[2*nb+1,st+1] = ctmp[]
                LAPACK.larfg!(len, @AU_ptr(st, st), pointer(VQ, vpos+1), Ref(TAUQ, taupos))

                lenj = len-1;
                ctmp[] = conj(TAUQ[taupos]);

                # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'L',
                #          len, lenj, VQ(vpos), ctmp, AU(st, st+1), LDX, WORK);
                # LAPACK.larf!('L', len, lenj, pointer(VQ, vpos),
                #     ctmp[], pointer(A, 2*nb+lda*(st+1)), LDX, WORK)
                LAPACK.larf!('L', len, lenj, pointer(VQ, vpos),
                    ctmp[], @AU_ptr(st, st+1), LDX, WORK)


            else 
                # /* ========================
                #  *       LOWER CASE
                #  * ========================*/
                # // Eliminate the col  at st-1

                VQ[vpos] = 1.
                # memcpy( VQ(vpos+1), AL(st+1, st-1), (len-1)*sizeof(coreblas_complex64_t) );
                # memset( AL(st+1, st-1), 0, (len-1)*sizeof(coreblas_complex64_t) );
                # VQ[vpos+1:vpos+len-1] .= A[nb+3:nb+len+1,st]
                # A[nb+3:nb+len+1,st] .= 0
                VQ[vpos+1:vpos+len-1] .= @ALv_(st+1, st-1, len-1)
                @ALv_set(st+1, st-1, 0, len-1)

                # LAPACKE_zlarfg(len, AL(st, st-1), VQ(vpos+1), 1, TAUQ(taupos) )
                # ctmp[] = A[nb+2,st]
                # LAPACK.larfg!(len, ctmp, pointer(VQ, vpos+1), Ref(TAUQ, taupos))
                # A[nb+2,st] = ctmp[]
                LAPACK.larfg!(len, @AL_ptr(st, st-1), pointer(VQ, vpos+1), Ref(TAUQ, taupos))

                # // Apply left on A(st:ed,st:ed) 
                ctmp[] = conj(TAUQ[taupos]);

                # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'L',
                #             len, len, VQ(vpos), ctmp, AL(st, st), LDX, WORK);
                # LAPACK.larf!('L', len, len, pointer(VQ, vpos),
                #             ctmp[], pointer(A, nb+1+st*lda), LDX, WORK)
                LAPACK.larf!('L', len, len, pointer(VQ, vpos),
                            ctmp[], @AL_ptr(st, st), LDX, WORK)

                # // Eliminate the created row at st 
                VP[vpos] = 1.;

                for i in 1:(len-1)
                    # VP[vpos+i] = conj(A[nb+1-i,st+1+i])
                    # A[nb+1-i,st+1+i] = 0.
                    VP[vpos+i] = conj(@AL(st, st+i))
                    @AL_set(st, st+i, 0.)
                end
                # ctmp[] = conj(A[nb+1,st+1])
                ctmp[] = conj(@AL(st, st))
                
                # LAPACKE_zlarfg(len, &ctmp, VP(vpos+1), 1, TAUP(taupos) );
                LAPACK.larfg!(len, ctmp, pointer(VP, vpos+1), Ref(TAUP, taupos))
                # A[nb+1,st+1] = ctmp[]
                @AL_set(st, st, ctmp[])

                lenj = len-1;
                ctmp = TAUP[taupos]
                # LAPACKE_zlarfx(LAPACK_COL_MAJOR, 'R',
                #                 lenj, len, VP(vpos), ctmp, AL(st+1, st), LDX, WORK);
                # LAPACK.larf!('R', lenj, len, pointer(VP, vpos),
                #             ctmp[], pointer(A, nb+2+st*lda), LDX, WORK)
                LAPACK.larf!('R', lenj, len, pointer(VP, vpos),
                            ctmp[], @AL_ptr(st+1, st), LDX, WORK)

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
