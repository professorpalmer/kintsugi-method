# data/

Licences and sources: `../LICENSES/DATA.md`. Names of arms and probes are the names of the project record; Appendix A
of the paper maps them to the names used in the text (for example V5bc-h32 = "KL-only winner", V5c = "KL-table B").

| Folder | Content | Read by |
|---|---|---|
| `appworld/` | `<arm>-s<seed>.csv`: `task_id, success` (1 / 0) per task-run of AppWorld test_normal (168 tasks per seed). Arms: `v3ac`, `kship`, `pv3ac` = the shipped file (selection, fresh 102 + 103, fresh 104 + 105); `kfw` = its weights with f16 K/V on an 80 GB card; `bf16o`, `bf16f`, `pbf16` = BF16; `v5bc` / `pv5bc`, `v5t` / `pv5tc`, `v2body` / `pv2body` = the candidates; `q2kxlo`, `pnaive` = the naive UD-Q2_K_XL file in our engine; Inkling 2 Mini: `inkm` = M2 (ours), `raw` = the official weights, `q65` / `q262` = M2 with the 4-bit global cache, `ink` = Unsloth's setup; `t3-pv3ac`, `t3-pv5bc` = the shipped file and the KL-only winner on AppWorld test_challenge (417 tasks per seed, seeds 106 + 107). A file without a seed suffix (`v3ac`, `v5bc`, `v2body`) is seed 100 | `scripts/gg_stats.py`, `scripts/make_figs.py` |
| `margins/gc2/` | Inkling graded K/V margins: arms a (release), b (blanket q4_0), c (blanket q8_0), d (4-bit global cache) at 117k / 245k; `repeat/` = arm a repeated; `rig/` = arm a on a second machine (cross-check only) | `scripts/gc2_score.py` |
| `margins/gb2/` | the 44 recall items at 117k for the shipped file and the two baselines | `scripts/gb2_score.py` |
| `b1/out/`, `b1c/out/` | B1 and B1c margin records, one file per arm and document; `repeat/` = the run-floor repeat | `scripts/b1/b1_score.py`, `scripts/b1/b1c_score.py` |
| `b1/population.jsonl.gz` | the B1 population (sha256 bc46d5dabc423d33b86c06c63c1f1260614eeb458c5ec2af523be4517aaba337) | `scripts/b1/b1_probe.py` |
| `bench/pycode.raw` | the copy corpus of the recall gate | `scripts/bench/` |
| `kl/<p1/p2>-<name>.csv` | per document: category, scored rows, KL sum, top-1 sum (p1 = build half, p2 = untouched half; `box-` = measured on the rented box with its own shipped-file reference) | `scripts/kl/kl_docs.py` |
