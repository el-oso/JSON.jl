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
    buf = Vector{UInt8}(codeunits(repeat("a", 256) * "\""))
    n = length(buf)
    GC.@preserve buf begin
        p = pointer(buf)
        JSON._string_scan_simd(p, 1, n)                       # warm up codegen
        # The classifier must lower to a vector compare (<64 x i8> in the LLVM IR).
        @assert_vectorized JSON._string_scan_simd(p, 1, n)
        # And it must run without heap allocation …
        @assert_noalloc    JSON._string_scan_simd(p, 1, n)
        # … and be type-stable (returns Int).
        @assert_typestable JSON._string_scan_simd(p, 1, n)
        # No scalar hot loop "leaks" between the vector work. NOTE (StrictMode feedback): this PASSES
        # even though the kernel has a bounded scalar tail — `scalar_fp_loops` is function-level (it
        # flags a scalar loop only when NO `<N x …>` op is present anywhere). Since the main loop emits
        # `<64 x i8>`, the tail isn't flagged. Good here (the tail is < 64 iters, LLVM handles it), but
        # it means the guarantee wouldn't catch a scalar tail inside an otherwise-vectorized function.
        @assert_no_scalar_loops JSON._string_scan_simd(p, 1, n)
    end
end
