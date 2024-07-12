using LinearAlgebra
using BenchmarkTools
using Plots

include("zlarfb_v3.jl")
include("zlarfbalt.jl")

function zunmqrv0(side, trans, m, n, k, ib, A, lda, T, ldt, C, ldc, work, ldwork)
	
	if side != 'L' && side != 'R'
        throw(ArgumentError("illegal value of side"))
		return -1
	end

	if side == 'L'
		nq = m
		nw = n
	else
		nq = n
		nw = m
	end

    #do we need to include 'T' since ConjTrans is technically 'C'  ?
	if trans != 'N' && trans != 'C' && trans != 'T'
        throw(ArgumentError("illegal value of trans"))
		return -2
	end

	if m < 0
        throw(ArgumentError("illegal value of m"))
		return -3
	end

	if n < 0
        throw(ArgumentError("illegal value of n"))
		return -4
	end

	if k < 0 || k > nq
        throw(ArgumentError("illegal value of k"))
		return -5
	end

	if ib < 0 
        throw(ArgumentError("illegal value of ib"))
		return -6
	end

	if lda < max(1, nq) && nq > 0
        throw(ArgumentError("illegal value of lda"))
		return -8
	end

	if ldt < max(1,ib)
        throw(ArgumentError("illegal value of ldt"))
		return -10
	end

	if ldc < max(1,m) && m > 0
        throw(ArgumentError("illegal value of ldc"))
		return -12
	end

	if ldwork < max(1,nw) && nw > 0 # so work is at least n x n or m x ib so as long as n >= ib it ok
        throw(ArgumentError("illegal value of ldwork"))
		return -14
	end

	# quick return 

	if m == 0 || n == 0 || k == 0
		return
	end

	if ((side == 'L' && trans != 'N') || (side == 'R' && trans == 'N'))
		i1 = 1
		i3 = ib
		ibstop = k
	else
		i1 = div((k-1),ib)*ib + 1
		i3 = -ib
		ibstop = 1
	end
	#println("start is ", i1, " step is ", i3, " stop is ", ibstop)
	
	ic = 1
	jc = 1
	ni = n
	mi = m

	if side == 'L'
		wwork = ones(eltype(A), n, ib)
		ldw = n
	else
		wwork = ones(eltype(A), m, ib)
		ldw = m
	end

	for i in i1 : i3 : ibstop # start, step, stop
		kb = min(ib, k-i+1)
		#println("at i = ", i, " kb is ", kb)

		if side == 'L'
			# apply to C[i:m, 1:n]
			mi = m - i + 1
			ic = i
		else
			# apply to C[1:m, i:n]
			ni = n-i + 1
			jc = i
		end

        cv = @view C[ic:m, jc:n]

        zlarfbv3(side, trans, 'F', 'C', mi, ni, kb, (@view A[i:lda, i:i+kb-1]), lda-i+1, (@view T[1:kb, i:i+kb-1]), kb, cv, ldc, (@view wwork[:, 1:kb]), ldw)
        #zlarfbalt(side, trans, 'F', 'C', mi, ni, kb, A,i,i, (@view T[1:kb, i:i+kb-1]), C,ic, jc , (@view work[:, 1:kb]), ldwork)
	end
end