# Plot for string_scan_bench.md (data inline from the committed run; regenerate the PNG with any
# Plots.jl-equipped project):  julia --project=<env-with-Plots> perf/plot_string_scan.jl
using Plots
const L      = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512]
const stock  = [0.595, 0.730, 0.836, 1.008, 1.025, 1.196, 1.391, 1.644, 1.612, 1.735, 2.210, 2.623]
const branch = [0.606, 0.719, 0.824, 0.765, 0.915, 0.937, 1.341, 1.785, 2.513, 3.248, 5.960, 10.710]
const simdj  = [0.353, 0.428, 0.970, 1.106, 1.299, 1.495, 1.723, 1.888, 2.272, 2.444, 2.883, 3.117]

p = plot(; xscale = :log2, xlabel = "string-field length (bytes)", ylabel = "isvalidjson GB/s (higher = better)",
    title = "JSON.jl stage-1 SIMD string scan vs stock & simd-json (4 MiB, single-thread)",
    titlefontsize = 10, legend = :topleft, framestyle = :box, dpi = 200, size = (920, 560),
    xticks = (L, string.(L)))
plot!(p, L, stock;  label = "stock JSON.jl",        lw = 2, marker = :circle,   color = :seagreen)
plot!(p, L, branch; label = "this branch (SIMD)",   lw = 2, marker = :diamond,  color = :steelblue)
plot!(p, L, simdj;  label = "simd-json (Rust tape)", lw = 2, marker = :utriangle, color = :slategray)
vspan!(p, [16, 48]; color = :red, alpha = 0.07, label = "regression valley")
savefig(p, joinpath(@__DIR__, "string_scan_bench.png"))
println("wrote perf/string_scan_bench.png")
