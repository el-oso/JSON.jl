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
#
# POC result (4 MiB doc, single-thread):
#   * kernel alone: ~28× the scalar byte loop at finding string boundaries (80 vs 2.8 GB/s).
#   * end-to-end `isvalidjson` is WORKLOAD-DEPENDENT (vs the simd-json Rust crate's tape build):
#       long strings (200–600 ch):  ~5.6 GB/s  vs simd-json 1.5  → ~3.7× FASTER.
#       short strings (typical):    within ~6% of stock JSON.jl (0.616 vs 0.654) — see below.
#   The short-string story took two wrong turns before landing: (1) a 64-byte SIMD load reads a fixed
#   window per string, and for closely-spaced short strings consecutive windows OVERLAP → each byte read
#   3–5× (read amplification: SIMD 0.45× scalar at ~12-byte strings, crossover ~24 — streaming probe).
#   (2) routing every string through an inline SIMD scan also hoisted its constant setup to the per-string
#   path. FIX: `parsestring` keeps its ORIGINAL tight scalar loop for the first `_SCALAR_WINDOW` bytes and
#   only escapes to the OUT-OF-LINE `_scan_wide` (`@noinline`) for long strings — so short strings touch
#   no SIMD machinery (regression 14% → ~6%; the residual is the per-byte window check) and long strings
#   keep the win. `_scan_wide`'s scalar tail is flagged by StrictMode F32 (the per-loop
#   `@assert_no_scalar_loops` this POC prompted) and accepted as a bounded epilogue (see test/).
using SIMD: Vec, vload, bitmask

const _SIMD_W = 64   # Vec{64,UInt8}: full AVX-512 zmm width
# Scalar fast-path window. A 64-byte SIMD load reads a fixed 64-byte window per string; for short
# strings spaced a few bytes apart (typical JSON fields), consecutive windows OVERLAP, so each byte is
# read 3–5× (measured: SIMD 0.45× scalar at ~12-byte strings, crossover ~24 — see the streaming probe).
# So we scalar-scan this many bytes first (each byte read once, no SIMD setup) and only escape to the
# wide SIMD scan for genuinely long strings, where the 64-byte window amortizes.
const _SCALAR_WINDOW = 32

# A byte ends a string-content run iff it is '"' (0x22), '\\' (0x5C), or a control byte (< 0x20).
@inline _is_string_boundary(b::UInt8) = (b == 0x22) | (b == 0x5C) | (b < 0x20)

# Wide SIMD scan for LONG strings. `@noinline` is load-bearing: it keeps the three vector-constant
# broadcasts + the SIMD loop OUT of the hot per-string path, so a short string that resolves in the
# scalar window never materializes any SIMD machinery (the bug that sank the earlier inline attempt).
@noinline function _scan_wide(p::Ptr{UInt8}, pos::Int, len::Int)
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

# Entry: `p` points at buffer byte 1 (1-based). Return the 1-based index in `pos:len` of the first
# string-boundary byte, or `len + 1` if none. Short strings stay scalar (no over-read, no SIMD setup);
# long strings escape to the out-of-line `_scan_wide`. Reads are bounded by `len` (no over-read).
@inline function _string_scan_simd(p::Ptr{UInt8}, pos::Int, len::Int)
    stop = min(pos + _SCALAR_WINDOW - 1, len)
    i = pos
    @inbounds while i <= stop
        _is_string_boundary(unsafe_load(p, i)) && return i
        i += 1
    end
    i > len && return len + 1            # window hit EOF without a boundary
    return _scan_wide(p, i, len)         # string longer than the window → wide SIMD
end

# Long-string fast-forward, called by `parsestring` ONCE a string has exceeded the scalar window.
# Contiguous byte buffers (Vector{UInt8} / String / CodeUnits) jump to the next boundary via the
# out-of-line SIMD scan; for any other (exotic) AbstractVector{UInt8}/AbstractString it is a no-op that
# returns `pos` — the caller's scalar loop keeps advancing one byte at a time (correct, just not SIMD).
@inline _scan_fwd(buf::Union{Vector{UInt8},Base.CodeUnits{UInt8,String},String}, pos::Int, len::Int) =
    GC.@preserve buf _scan_wide(pointer(buf isa Base.CodeUnits ? buf.s : buf), pos, len)
@inline _scan_fwd(buf, pos::Int, len::Int) = pos
