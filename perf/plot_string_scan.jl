# Plot for string_scan_bench.md (data inline from the committed run; regenerate the PNG with any
# Plots.jl-equipped project):  julia --project=<env-with-Plots> perf/plot_string_scan.jl
using Plots
# Clean same-core run (core 11, σ ≤ 8%), inline-leaf parsestring.
const L      = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512]
const stock  = [0.475, 0.571, 0.683, 0.772, 0.786, 0.914, 1.097, 1.242, 1.239, 1.296, 1.671, 1.990]
const branch = [0.426, 0.531, 0.602, 0.647, 0.780, 0.778, 1.172, 1.486, 2.061, 2.672, 4.871, 8.790]
const simdj  = [0.278, 0.342, 0.753, 0.867, 1.013, 1.161, 1.349, 1.468, 1.789, 1.934, 2.293, 3.119]

p = plot(; xscale = :log2, xlabel = "string-field length (bytes)", ylabel = "isvalidjson GB/s (higher = better)",
    title = "JSON.jl stage-1 SIMD string scan vs stock & simd-json (4 MiB, single-thread)",
    titlefontsize = 10, legend = :topleft, framestyle = :box, dpi = 200, size = (920, 560),
    xticks = (L, string.(L)))
plot!(p, L, stock;  label = "stock JSON.jl",        lw = 2, marker = :circle,   color = :seagreen)
plot!(p, L, branch; label = "this branch (SIMD)",   lw = 2, marker = :diamond,  color = :steelblue)
plot!(p, L, simdj;  label = "simd-json (Rust tape)", lw = 2, marker = :utriangle, color = :slategray)
vspan!(p, [4, 44]; color = :red, alpha = 0.07, label = "branch < stock (≤ ~40 B)")
savefig(p, joinpath(@__DIR__, "string_scan_bench.png"))
println("wrote perf/string_scan_bench.png")
