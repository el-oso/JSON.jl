# Stage-1 SIMD string scanner — size-sweep benchmark

Reference numbers for the `strictmode-simd-stage1` branch: `parsestring`'s byte-by-byte scan for the
end of a string is replaced by a SIMD classifier (`Vec{64,UInt8}` → `<64 x i8>` AVX-512 `vpcmpb` + mask)
for long strings, while keeping the original scalar loop for the common short-string case.

**Metric:** `JSON.isvalidjson(bytes)` throughput (GB/s) — a full structural pass that exercises
`parsestring` heavily, with zero output allocation (so the number is the parse work, not GC). Documents
are 4 MiB arrays of `{id, a, b, c}` records whose three string fields are each exactly `strlen` bytes
(uniform `a–z`, no escapes), regenerated deterministically per size.

**Setup:** single-thread, `julia -O3 -t1`, `taskset -c 4` (SMT sibling idle), `RAYON_NUM_THREADS=1`,
median of 400 reps with `GC.gc()` between and a DCE sink. `simd-json` = the Rust crate's `to_tape`
(via the BlazingPorts `bp_simdjson_parse` shim) on the *same* bytes — it must copy the input first (it
unescapes in place), which is ~1% of its time. Machine: Zen5, 4.5 GHz. Reproduce with
[`string_scan_bench.jl`](string_scan_bench.jl).

![isvalidjson GB/s vs string length](string_scan_bench.png)

(Regenerate the plot from the saved data with [`plot_string_scan.jl`](plot_string_scan.jl).)

## Results (GB/s, higher is better)

| string len | stock JSON.jl | this branch | simd-json (Rust) | branch / stock | branch / simd-json |
|-----------:|--------------:|------------:|-----------------:|---------------:|-------------------:|
|   4 | 0.595 |  0.606 | 0.353 | 1.02× | **1.72×** |
|   8 | 0.730 |  0.719 | 0.428 | 0.99× | **1.68×** |
|  12 | 0.836 |  0.824 | 0.970 | 0.99× | 0.85× |
|  16 | 1.008 |  0.765 | 1.106 | **0.76×** | 0.69× |
|  24 | 1.025 |  0.915 | 1.299 | 0.89× | 0.70× |
|  32 | 1.196 |  0.937 | 1.495 | **0.78×** | 0.63× |
|  48 | 1.391 |  1.341 | 1.723 | 0.96× | 0.78× |
|  64 | 1.644 |  1.785 | 1.888 | 1.09× | 0.95× |
|  96 | 1.612 |  2.513 | 2.272 | 1.56× | 1.11× |
| 128 | 1.735 |  3.248 | 2.444 | 1.87× | 1.33× |
| 256 | 2.210 |  5.960 | 2.883 | 2.70× | 2.07× |
| 512 | 2.623 | 10.710 | 3.117 | 4.08× | **3.44×** |

(`simd-json` column from the branch run; the stock run measured it within run-to-run noise, ≤10%.)

## Reading the curve

- **Long strings (≥ 64 B): a clear, growing win.** The branch overtakes stock at ~64 B and reaches
  **4.1× stock / 3.4× simd-json at 512 B** — the SIMD classifier scans content at memory-bandwidth speed
  while the scalar loop plods byte-by-byte. This is the regime the change is for (text blobs, base64,
  long descriptions, embedded documents).
- **Very short strings (≤ 12 B): parity** with stock. They never leave the scalar fast-path, so they
  cost the same (here the branch even edges out simd-json, whose fixed per-document tape overhead
  dominates when records are tiny).
- **Medium strings (16–48 B): a regression valley** — the branch is **0.76–0.96× stock** (worst ~24% at
  16 B). These strings are too long to finish inside the scalar window but too short to amortize the
  64-byte SIMD window, and they still pay the per-byte "have I passed the window yet?" check that the
  original tight loop doesn't have. This is the honest cost of the current `_SCALAR_WINDOW = 32` split.

**Net:** a strong win when strings are long, parity when they're tiny, and a medium-string valley. Whether
that trade is worth it depends on the workload — string-heavy/long-field JSON benefits a lot; uniformly
short-field JSON (many 16–32 B keys/values) regresses. The valley is the obvious tuning target:
`_SCALAR_WINDOW` and the per-byte window check (e.g. a branchless or unrolled prefix) both move it.

## Caveats

- `isvalidjson` validates structure; it does not build a Julia value graph. Full `JSON.parse` adds the
  same allocation on both sides, so the *relative* picture holds but absolute GB/s is lower and GC-noisier.
- `simd-json` builds a usable tape; `isvalidjson` does not — so the simd-json column is a generous
  comparison (it does strictly more work), which makes the branch's ≥ 96 B lead the more notable.
- Numbers are single-core; none of the three uses threads here.
