# Data in this repository: sources and terms

| Folder / file | What it is | Source and terms |
|---|---|---|
| `data/appworld/*.csv` | outcome of every AppWorld task-run the paper uses: `task_id, success` (one file per arm and seed) | our runs of the AppWorld benchmark (Trivedi et al., ACL 2024; https://github.com/StonyBrookNLP/appworld). Only task ids and pass / fail are given: no task text, no environment data, no test assertions. AppWorld itself is under its own licence; use these outcomes with it under its terms. Our outcome files: CC BY 4.0 |
| `data/margins/`, `data/b1/out/`, `data/b1c/out/` | teacher-forced margin records of our model files (per item: margins in nats, item names, depth) | our measurements; CC BY 4.0. The item texts themselves are built by the scripts from the files below |
| `data/b1/population.jsonl.gz` | the B1 recall population: haystacks of Python source and the questions / answers built from them | derived from the CPython 3.12.10 standard library: PSF License 2.0 (`LICENSES/PSF-2.0.txt`); our selection and questions: CC BY 4.0 |
| `data/b1/excluded_names.txt` | file names left out of the B1 pool (the selection's recall corpora) | CC BY 4.0 |
| `data/bench/pycode.raw` | the copy corpus of the recall gate (Python source files in order) | derived from the CPython standard library: PSF License 2.0 |
| `data/kl/*.csv` | per-document sums of our KL measurements on the Qwen3.8-27B quantization challenge's public golden set | our measurements: CC BY 4.0. The golden set (CC BY 4.0, by StillDeadcode) is not redistributed; the document index refers to its public half |

Model weights are not in this repository; see the two model repositories linked in `README.md` for their licences.
