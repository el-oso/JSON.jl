# StrictMode coverage for the stage-1 SIMD string classifier (POC).
# JSON.jl source carries NO StrictMode dependency (the SIMD kernel is plain SIMD.jl); the formal
# guarantees on that kernel live here, in a dedicated test environment — mirroring the BlazingPorts
# discipline. This is the real-world probe: does StrictMode confirm the kernel vectorizes / stays
# allocation-free, and what does it flag? Run:
#   julia --project=test/strictmode test/strictmode/runtests.jl
using Test, SIMD, Random
import JSON
using StrictMode, AllocCheck, JET

# scalar reference with identical semantics to _is_string_boundary (the oracle for parity)
function scan_ref(buf, pos, len)
    @inbounds while pos <= len
        b = buf[pos]
        (b == 0x22 || b == 0x5c || b < 0x20) && return pos
        pos += 1
    end
    return len + 1
end

@testset "stage-1 SIMD classifier — parity vs scalar (random + boundaries)" begin
    Random.seed!(0xC0FFEE)
    for _ in 1:5000
        n = rand(1:300)
        buf = rand(UInt8, n)
        for _ in 1:rand(0:6)                      # sprinkle boundary bytes at random spots
            buf[rand(1:n)] = rand((0x22, 0x5c, 0x00, 0x1f, 0x09, 0x0a))
        end
        GC.@preserve buf begin
            p = pointer(buf)
            for start in unique((1, n, rand(1:n), rand(1:n)))
                @test JSON._string_scan_simd(p, start, n) == scan_ref(buf, start, n)
            end
        end
    end
    # explicit boundary sizes around the 64-byte SIMD block (off-by-one nightmares)
    for n in (1, 63, 64, 65, 127, 128, 129)
        buf = fill(UInt8('a'), n); buf[n] = 0x22  # quote exactly at the end
        GC.@preserve buf begin
            p = pointer(buf)
            @test JSON._string_scan_simd(p, 1, n) == scan_ref(buf, 1, n) == n
        end
    end
end

@testset "stage-1 SIMD classifier — StrictMode guarantees" begin
    # The SIMD work lives in `_scan_wide` (the out-of-line long-string kernel); `_string_scan_simd` /
    # `parsestring` keep a scalar fast-path for short strings, so the vector guarantees target `_scan_wide`.
    buf = Vector{UInt8}(codeunits(repeat("a", 256) * "\""))
    n = length(buf)
    GC.@preserve buf begin
        p = pointer(buf)
        JSON._scan_wide(p, 1, n)                              # warm up codegen
        @assert_vectorized JSON._scan_wide(p, 1, n)           # <64 x i8> in the LLVM IR
        @assert_noalloc    JSON._scan_wide(p, 1, n)           # no heap allocation
        @assert_typestable JSON._scan_wide(p, 1, n)           # returns Int
    end
end

@testset "stage-1 SIMD classifier — F32 dogfood (per-loop @assert_no_scalar_loops)" begin
    # This is the StrictMode finding from this POC, fed back and FIXED (StrictMode F32). `_scan_wide` has
    # a SIMD main loop (`<64 x i8>`) AND a bounded scalar tail loop. BEFORE the fix, `scalar_fp_loops`
    # short-circuited on `_vectorized(f) && return false`, so it was a FALSE-NEGATIVE — the scalar tail
    # was invisible and `@assert_no_scalar_loops` wrongly passed. AFTER the per-loop fix it correctly
    # SEES the tail. We exercise the fixed behavior here:
    @test StrictMode.scalar_fp_loops(JSON._scan_wide, (Ptr{UInt8}, Int, Int)) == true
    buf = Vector{UInt8}(codeunits(repeat("a", 256) * "\""))
    n = length(buf)
    GC.@preserve buf begin
        p = pointer(buf)
        JSON._scan_wide(p, 1, n)
        @test_throws StrictViolation (@assert_no_scalar_loops JSON._scan_wide(p, 1, n))
    end
    # The tail is ACCEPTED (a bounded < 64-byte remainder, like an auto-vectorization epilogue) — F32's
    # value is making it VISIBLE so the decision is explicit, not silently hidden by a coexisting `<N x>`.
end
