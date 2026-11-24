# Per-call seeds for AppWorld runs (a patch proposal)

Status: a proposal. No run of the paper used it. It is not tested. The finding that makes it necessary is row E5.1
of `EVIDENCE.md` and Section 4.1 of the paper.

## The fault

The AppWorld client puts the run's seed in the model configuration. It sends that same value with every chat
request. A llama.cpp-type server makes a new sampler for each request and seeds it with the value of the request. So
every model call of every task of a run starts from the same random stream, and two arms on the same seed also share
it. The task-runs of one seed are then not independent draws. In our data the first reply began with the same 8 words
on 51.2 % of the tasks for two different bit maps on the same seed, and on 23.8 % for two maps on different seeds
(`results/gg-seed.txt`, part 1b).

A test that treats the task-runs as independent (for example an exact sign test over all task-runs) gives p-values
that are too small. The paper uses a design-effect test and a cluster-robust interval for the runs that exist
(`scripts/gg_seed.py`).

## The change

Derive the seed of each request from three values: the run's seed, the task id and the number of the call in the
task. Then:

- the draws are independent across tasks and across the calls of one task;
- a run stays reproducible (the same three values give the same seed);
- two arms keep the same draw on the same task and call, so paired comparisons stay sharp.

The cleanest place is the client, at the one model call of the ReAct agent. In the AppWorld source that we ran, this
call is in `src/experiments/code/simplified/react_code_agent.py`:

```python
output = self.language_model.generate(messages=messages, cache_control_at=-1)
```

Proposed form (the function `generate` already merges call arguments over the model configuration, so no other
change is necessary, and the per-call seed is written to the call log with the request):

```python
import hashlib

def derive_seed(base: int, task_id: str, step: int) -> int:
    """A reproducible 31-bit seed for one model call. Never 0xFFFFFFFF, which llama.cpp reads as "random seed"."""
    digest = hashlib.sha256(f"{base}:{task_id}:{step}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

# at the model call of the agent:
base = self.language_model.generation_kwargs["seed"]          # the run's seed, as now (name of the field may differ)
output = self.language_model.generate(
    messages=messages,
    cache_control_at=-1,
    seed=derive_seed(base, self.world.task_id, self.step_number),
)
```

The names `generation_kwargs`, `self.world.task_id` and `self.step_number` are from our reading of the source at the
version that we ran. Check them against your version before you use the change.

## The other place (not preferred)

The server can mix a hash of the prompt into the seed of the request. That works for every client. It changes the
engine, and with it the identity checks of an engine release, so we do not propose it.

## What stays necessary

Per-call seeds remove the shared random stream. They do not remove the need for seeds as units: use three or more
seeds for each comparison, give the difference on each seed, and test at the seed level. A temperature-0 run has no
random stream and is a useful control, but it is one draw of the model's behaviour.
