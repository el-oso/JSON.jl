# ── Stage-1 SIMD structural classifier (POC) ─────────────────────────────────────────────────────────
# simdjson's key idea is a "stage 1" that classifies many bytes at once into structural / interesting
# positions, so the parser jumps between them instead of inspecting every byte scalar-ly. JSON.jl's
# dominant byte-by-byte loop is `parsestring`'s scan for the end of a string: it walks one byte at a
# time looking for the closing '"', a '\\' escape, or a raw control byte (< 0x20). This kernel does that
# classification 64 bytes at a time with `Vec{64,UInt8}` (→ `<64 x i8>` AVX-512 `vpcmpb` + mask), then
# `trailing_zeros` on the lane bitmask to land on the first interesting byte.
#
# Pure SIMD.jl — the source carries NO StrictMode dependency (mirrors the BlazingPorts discipline); the
# `@assert_vectorized` / `@assert_noalloc` guarantees on this kernel live in `test/` only.
using SIMD: Vec, vload, bitmask

const _SIMD_W = 64   # Vec{64,UInt8}: full AVX-512 zmm width

# A byte ends a string-content run iff it is '"' (0x22), '\\' (0x5C), or a control byte (< 0x20).
@inline _is_string_boundary(b::UInt8) = (b == 0x22) | (b == 0x5C) | (b < 0x20)

# SIMD core: `p` points at buffer byte 1 (1-based). Return the 1-based index in `pos:len` of the first
# string-boundary byte, or `len + 1` if none. Reads are bounded by `len` (no padding / over-read).
@inline function _string_scan_simd(p::Ptr{UInt8}, pos::Int, len::Int)
    vquote = Vec{_SIMD_W,UInt8}(0x22)
    vback  = Vec{_SIMD_W,UInt8}(0x5C)
    vctrl  = Vec{_SIMD_W,UInt8}(0x20)   # control bytes are strictly below this (0x20 = space, valid)
    i = pos
    @inbounds while i + _SIMD_W - 1 <= len
        c  = vload(Vec{_SIMD_W,UInt8}, p + (i - 1))      # bytes i .. i+63
        m  = (c == vquote) | (c == vback) | (c < vctrl)  # lane-wise boundary mask
        bm = bitmask(m)
        bm != zero(bm) && return i + trailing_zeros(bm)  # first boundary lane
        i += _SIMD_W
    end
    @inbounds while i <= len                              # scalar tail (< 64 bytes from the end)
        _is_string_boundary(unsafe_load(p, i)) && return i
        i += 1
    end
    return len + 1
end

# Dispatch: contiguous byte buffers (Vector{UInt8} / String) take the SIMD path; anything else (exotic
# AbstractVector{UInt8}/AbstractString) falls back to a scalar scan with identical semantics.
@inline function _string_scan(buf::Union{Vector{UInt8},Base.CodeUnits{UInt8,String},String}, pos::Int, len::Int)
    GC.@preserve buf (_string_scan_simd(pointer(buf isa Base.CodeUnits ? buf.s : buf), pos, len))
end
@inline function _string_scan(buf, pos::Int, len::Int)
    @inbounds while pos <= len
        _is_string_boundary(getbyte(buf, pos)) && return pos
        pos += 1
    end
    return len + 1
end
