# Size-sweep benchmark for the stage-1 SIMD string scanner (branch `strictmode-simd-stage1`).
#
# Measures JSON.jl `isvalidjson` throughput (a full structural pass that exercises `parsestring`) across
# a range of string-field lengths, and — when the simd-json comparison shim is available — the simd-json
# Rust crate's tape build on the *same* documents. Run it once with the SIMD branch and once with stock
# JSON.jl to fill both JSON columns:
#
#   RAYON_NUM_THREADS=1 taskset -c 4 julia -O3 -t1 --project=. perf/string_scan_bench.jl        # branch
#   RAYON_NUM_THREADS=1 taskset -c 4 julia -O3 -t1 --project=/path/to/stockenv perf/string_scan_bench.jl
#
# The simd-json column needs the BlazingPorts comparison cdylib (`libblazing_compare.so`, which exports
# `bp_simdjson_parse`). Point BLAZING_LIB at it; without it the simd-json column is omitted gracefully.
# Single-thread, pin the core (`taskset`) and idle its SMT sibling for low noise.
using Printf, Random
import JSON

const LIB = get(ENV, "BLAZING_LIB",
    joinpath(homedir(), "Documents/claude/BlazingPorts.jl/bench/rust_compare/rust/target/release/libblazing_compare.so"))
const HAVE_SJ = isfile(LIB)

# Document of `target` bytes: array of {id, three string fields each exactly `L` bytes}. Deterministic.
function gen_doc(target::Int, L::Int; seed::UInt=0x5113D90 % UInt)
    rng = Xoshiro(seed); io = IOBuffer(); print(io, "["); i = 0
    while io.size < target
        i > 0 && print(io, ",")
        JSON.print(io, (id = i,
            a = String(rand(rng, 'a':'z', L)),
            b = String(rand(rng, 'a':'z', L)),
            c = String(rand(rng, 'a':'z', L))))
        i += 1
    end
    print(io, "]"); return take!(io)
end

med(ts) = (s = sort(ts); s[cld(length(s), 2)])
relσ(ts) = (m = sum(ts) / length(ts); sqrt(sum(x -> (x - m)^2, ts) / length(ts)) / m)

function bench(f, nb; reps = 400)
    f()                                              # warm
    ts = Float64[]
    for _ in 1:reps; GC.gc(); push!(ts, @elapsed f()); end
    return (nb / med(ts) / 1e9, relσ(ts))            # (GB/s, rel-σ)
end

const SIZES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512)
@printf("# JSON v%s   simd-json shim: %s\n", pkgversion(JSON), HAVE_SJ ? "present" : "MISSING (column omitted)")
@printf("# CSV: strlen,doc_MiB,records,json_gbs,json_relsig,simdjson_gbs,simdjson_relsig\n")
for L in SIZES
    b = gen_doc(4 * 1024 * 1024, L); nb = length(b)
    recs = count(==(UInt8('{')), b)
    jg, js = bench(() -> (x = JSON.isvalidjson(b); Base.donotdelete(x); x), nb)
    if HAVE_SJ
        scratch = Vector{UInt8}(undef, nb + 64)
        sj = () -> GC.@preserve b scratch ccall((:bp_simdjson_parse, LIB), UInt64,
            (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, UInt32), b, nb, scratch, length(scratch), UInt32(1))
        sg, ss = bench(sj, nb)
        @printf("CSV,%d,%.2f,%d,%.3f,%.3f,%.3f,%.3f\n", L, nb / 1024^2, recs, jg, js, sg, ss)
    else
        @printf("CSV,%d,%.2f,%d,%.3f,%.3f,,\n", L, nb / 1024^2, recs, jg, js)
    end
    flush(stdout)
end
