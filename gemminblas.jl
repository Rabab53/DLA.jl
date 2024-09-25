using AMDGPU, BSON
n_values=(2 .^(1:14))
timings=zeros(4,length(n_values))

function mybelapsed(A, B)
   AMDGPU.rocBLAS.gemm('N','N',copy(A),copy(B))
   t=0.0
   k=0
   if(k<100 && t<0.1)
       Acpy=copy(A)
       Bcpy=copy(B)
       AMDGPU.synchronize(blocking=true)
       t+= @elapsed AMDGPU.@sync AMDGPU.rocBLAS.gemm('N','N',Acpy,Bcpy)
       AMDGPU.synchronize(blocking=true)
       AMDGPU.unsafe_free!(Acpy)
       AMDGPU.unsafe_free!(Bcpy)
       k+=1
    end
    GC.gc()
    return t/k
end


for (i,n) in enumerate(n_values)
   A=ROCArray(rand(Float32,n,n));
   B=ROCArray(rand(Float32,n,n));
   
   timings[1,i]=mybelapsed(A,B)
   @show n, timings[1,i]
   BSON.@save "AMD_matmul_bench.bson" timings
end
