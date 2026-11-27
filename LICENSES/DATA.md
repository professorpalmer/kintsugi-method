# Data in this repository: sources and terms

| Folder / file | What it is | Source and terms |
|---|---|---|
| `data/appworld/*.csv` | outcome of every AppWorld task-run the paper uses: `task_id, success` (one file per arm and seed) | our runs of the AppWorld benchmark (Trivedi et al., ACL 2024; https://github.com/StonyBrookNLP/appworld). Only task ids and pass / fail are given: no task text, no environment data, no test assertions. AppWorld itself is under its own licence; use these outcomes with it under its terms. Our outcome files: CC BY 4.0 |
| `data/margins/`, `data/b1/out/`, `data/b1c/out/` | teacher-forced margin records of our model files (per item: margins in nats, item names, depth) | our measurements; CC BY 4.0. The item texts themselves are built by the scripts from the files below |
| `data/b1/population.jsonl.gz` | the B1 recall population: haystacks of Python source and the questions / answers built from them | derived from the CPython 3.12.10 standard library: PSF License 2.0 (`LICENSES/PSF-2.0.txt`); our selection and questions: CC BY 4.0 |
| `data/b1/excluded_names.txt` | file names left out of the B1 pool (the selection's recall corpora) | CC BY 4.0 |
| `data/bench/pycode.raw` | the copy corpus of the recall gate (Python source files in order) | derived from the CPython standard library: PSF License 2.0 |
| `data/appworld/openings/*.csv` | per task-run: the first word of the model's first reply and a 16-character hash of its first 8 words | our runs; the model's own words, one word per task-run; no AppWorld task text. CC BY 4.0 |
| `data/habits/*.csv` | per-unit outcomes of the decision-point habit studies of paper v1.3: entry or task ids and 0 / 1 flags (and the prompt token count of a unit) | our runs of BFCL v3 multi-turn (the Gorilla project, Patil et al.; https://github.com/ShishirPatil/gorilla; Apache-2.0) and of AppWorld. Only ids and outcomes are given: no prompt, no reply, no tool definition, no test assertion. Our outcome files: CC BY 4.0 |
| `data/w2c/out/`, `data/margins/w2d/` | teacher-forced margin records of the second batch | our measurements; CC BY 4.0 |
| `data/freegen/` | free-generation records on a synthetic needle document (built by `scripts/free_needles.py`): the model's answer and its thinking text | our runs; the document is synthetic filler with planted codes; CC BY 4.0 |
| `data/scores/` | result tables of our scorers | our measurements; CC BY 4.0. `w4l-overlap.txt` and `w4l-old-file-clean-set.txt` report measurements on a third party's public files and name public datasets; no third-party text is included |
| `data/kl/*.csv` | per-document sums of our KL measurements on the Qwen3.8-27B quantization challenge's public golden set | our measurements: CC BY 4.0. The golden set (CC BY 4.0, by StillDeadcode) is not redistributed; the document index refers to its public half |

Model weights are not in this repository; see the two model repositories linked in `README.md` for their licences.
