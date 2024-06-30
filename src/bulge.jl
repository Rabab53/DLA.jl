using Base: cld

@inline function coreblas_ceildiv(a::Int, b::Int)::Int
    return cld(a, b)
end

@inline function findVTpos(
    N::Integer,
    NB::Integer, 
    Vblksiz::Integer,
    sweep::Integer, 
    st::Integer) 
        
    prevcolblknb = 0
    prevblkcnt   = 0
    curcolblknb  = 0

    nbprevcolblk = sweep/Vblksiz;

    for prevcolblkid in 0:(nbprevcolblk-1)
        mastersweep  = prevcolblkid * Vblksiz;
        prevcolblknb = cld((N-(mastersweep+2)),NB);
        prevblkcnt   = prevblkcnt + prevcolblknb;
    end

    curcolblknb = cld((st-sweep),NB);
    blkid       = prevblkcnt + curcolblknb -1;
    locj        = sweep%Vblksiz;
    LDV         = NB + Vblksiz -1;
  
    myblkid = blkid;
    Vpos    = blkid*Vblksiz*LDV  + locj*LDV + locj;
    TAUpos  = blkid*Vblksiz + locj;
    Tpos    = blkid*Vblksiz*Vblksiz + locj*Vblksiz + locj;

    return Int(Vpos), Int(TAUpos), Int(Tpos), Int(myblkid)
end

@inline function findVTsiz(
    N::Int,
    NB::Int,
    Vblksiz::Int,
    blkcnt::Int,
    LDV::Int)
  
    blkcnt = 0;
    nbcolblk = coreblas_ceildiv((N-1),Vblksiz);

    for colblk in 0:(nbcolblk-1)
        mastersweep = colblk * Vblksiz;
        curcolblknb = coreblas_ceildiv((N-(mastersweep+2)),NB);
        blkcnt    = blkcnt + curcolblknb;
    end

    blkcnt = blkcnt +1;
    LDV = NB+Vblksiz-1;

    return blkcnt, LDV
end