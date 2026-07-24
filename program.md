# autoresearch — RibOrthrus

This is an experiment to have the LLM do its own research on the RibOrthrus
prediction head.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag.** Propose a tag based on today's date (for example,
   `jul24`). The branch `autoresearch/<tag>` must not already exist; this is a
   fresh run.
2. **Require a clean worktree.** Before creating the branch, stop if
   `git status --short` is non-empty. Ask the user to commit, stash, or remove
   their unrelated work. This is required because discarded experiments are
   restored with Git.
3. **Create the branch.** From the current `main` branch, run
   `git checkout -b autoresearch/<tag>`.
4. **Read the in-scope files.** Read these files for full context:
   - `README.md` — repository context.
   - `scripts/prepare_data.py`, `scripts/collators.py`, and `scripts/loss.py`
     — fixed dataset, collation, and evaluation/loss code. Do not modify.
   - `scripts/models.py` — editable prediction-head architecture.
   - `scripts/train.py` — editable training harness and hyperparameters.
5. **Verify data exists.** Confirm `data/` contains one or more `.ribo` files.
   For default (non-fine-tuning) training, also confirm `embeddings/` contains
   the precomputed embedding files. If either is missing, tell the human to
   prepare/provide the data; do not modify data preparation during this run.
6. **Initialize results.** Create `results.tsv` with only the header row below.
   Do not commit it.
7. **Confirm and go.** Confirm setup looks good. Once the user confirms,
   establish the baseline and begin experimentation.

## Experimentation

Each experiment runs as one Slurm job on the `gh` GPU partition. The training
script has a fixed 10-minute wall-clock budget (Lightning's `max_time`,
excluding startup and shutdown). Do not run `scripts/train.py` directly on a
login node and do not modify `train.slurm`.

```bash
mkdir -p logs
job_id=$(sbatch --parsable train.slurm)
job_id=${job_id%%;*}
echo "Submitted job ${job_id}"
```

Wait for the job to leave the Slurm queue, then use these files:

```text
logs/riborthrus_ft_<job_id>.out
logs/riborthrus_ft_<job_id>.err
```

The goal is simple: get the lowest final `val_loss`. Since the time budget is
fixed, model architecture, optimizer, scheduler, batch size, precision,
gradient accumulation, clipping, and all other training/model hyperparameters
are fair game.

**What you CAN do:**

- Modify `scripts/train.py` and `scripts/models.py` only.
- Change the prediction-head architecture, optimizer, scheduler, and training
  loop as long as the model still produces a tensor matching `y_true`.
- Use packages already available in the environment.

**What you CANNOT do:**

- Modify `scripts/prepare_data.py`, `scripts/collators.py`, `scripts/loss.py`,
  dependencies, data, embeddings, or the Slurm launcher.
- Change the evaluation target or compare scores from different protocols. Keep
  model type, seed, train split, read lengths, sequence length, and fine-tuning
  setting fixed for comparable experiments.
- Install packages, modify the validation-loss reporting contract, or run more
  than one GPU experiment at once.

`PredictionHead.input_projection` is currently `nn.Identity()`. Therefore
`tower_channels` must remain 512 unless the architecture also introduces a
projection from the 512 input channels to the new width.

VRAM is a soft constraint. Some increase is acceptable for meaningful loss
improvement, but avoid dramatic growth or out-of-memory runs.

**Simplicity criterion:** all else equal, prefer simpler code. A small loss
improvement is not worth substantial brittle complexity; an equally good or
better simplification is a win.

**The first run:** always run the unchanged baseline first.

## Output format

At the end of a successful run, `scripts/train.py` prints:

```text
AUTORESEARCH_METRIC val_loss=<number>
```

Extract it without flooding the context, after substituting the submitted job
ID:

```bash
grep '^AUTORESEARCH_METRIC val_loss=' logs/riborthrus_ft_<job_id>.out
```

## Logging results

After every experiment append one tab-separated row to `results.tsv`. Do not
commit this file. It has exactly these columns:

```text
commit	val_loss	status	description
```

Use a seven-character commit hash; use `0.000000` for crashes.
`status` is `keep`, `discard`, or `crash`. Descriptions must be short and must
not contain tabs.

Example:

```text
commit	val_loss	status	description
a1b2c3d	1.234567	keep	baseline
b2c3d4e	1.210000	keep	increase head learning rate
c3d4e5f	1.250000	discard	use GELU in residual block
d4e5f6g	0.000000	crash	double tower width caused OOM
```

## The experiment loop

The run occurs on the dedicated branch `autoresearch/<tag>`.

LOOP FOREVER:

1. Check the current retained commit and `results.tsv`.
2. For the first iteration, run the unchanged baseline. Otherwise, make one
   hypothesis-driven change directly in `scripts/train.py` or `scripts/models.py`.
3. Verify syntax: `python -m py_compile scripts/train.py scripts/models.py`.
4. Commit only the two editable source files with a concise experiment message.
5. Submit the fixed-budget Slurm job above. Use `squeue -j <job_id>` to poll;
   queued time does not count toward the experiment timeout.
6. After the job completes, read the metric using the `grep` command above. If
   it is absent, inspect `tail -n 50 logs/riborthrus_ft_<job_id>.out` and
   `tail -n 50 logs/riborthrus_ft_<job_id>.err` for the failure.
7. Append a TSV row.
8. If `val_loss` is lower than the best comparable retained result, keep the
   commit and advance the branch. If it is equal or worse, restore the prior
   retained commit with `git reset --hard <retained-commit>`.

**Timeout**: Once a job is running, training should take ~10 minutes (+ a few
seconds for startup and evaluation). If it remains running beyond 15 minutes,
use `scancel <job_id>`, log a crash, and restore the prior retained commit.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 6/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
