# Report — Lethe: UI feedback + duplicate prompt-load output (2026-10-20)

- **Branch:** `fix/ui-feedback`
- **Owner:** Lars
- **Status:** DONE — committed, not pushed (PM merges)
- **Scope:** `ui` path + shared logging hygiene. CLI output contract untouched.

## Summary

Both reported issues are fixed, plus the two items added mid-flight in
`ADDENDUM-2026-10-20-lethe-ui-feedback.md` (HF download visibility, friendly
invalid-model-id errors).

1. **Duplicate prompt loads** — `load_prompts` is now memoized per process, keyed
   on the resolved path (or dataset name). The `loaded N prompts` line fires only
   on a real cache miss. Callers get a fresh copy so mutation cannot poison the
   cache.
2. **"Looks hung"** — the model load, direction computation, baseline eval and
   reference-distribution capture now emit real sub-progress; genuinely unbounded
   steps (weight download, single demo generations) switch the bar to an animated
   spinner instead of holding a static percentage.
3. **Friendly errors** — Ollama-style `name:tag` and malformed ids are rejected
   with guidance before a job starts, and known `huggingface_hub` validation
   errors are translated.

## What changed (file:line)

### Issue A — de-duplicate prompt loading
- `src/abliterate/data.py:24` — module-level `_PROMPT_CACHE`.
- `src/abliterate/data.py:36` — `load_prompts` routes every source through
  `_cached(...)`; signature unchanged (`load_prompts(source, *, default_file)`).
- `src/abliterate/data.py:60` — `_cached(key, loader)`: loads once, returns
  `list(cached)` (a copy) on every call.
- `src/abliterate/data.py:71` — `_clear_prompt_cache()` for tests.
- Logging stays inside `_read_text_file` / `_load_known_dataset`, which now run
  only on a miss — so the log line is emitted exactly once per distinct source.

### Issue B — real progress for the slow phases
- `src/abliterate/progress.py` (new) — `ProgressCallback` type + best-effort
  `emit(on_progress, phase, fraction)`. `fraction=None` means indeterminate.
- `src/abliterate/model_utils.py:102` — `load_model_and_tokenizer(cfg, *,
  on_progress=None)`; emits `Resolving tokenizer` → `Loading tokenizer` →
  `Downloading weights`/`Reading weights from disk` (indeterminate) →
  `Loading into memory` → `Merging LoRA adapter` → `Resolving architecture` →
  `Model ready`.
- `src/abliterate/pipeline.py:53` — `build_directions(..., on_progress=None)`
  emits `Collecting harmful activations` → `Collecting harmless activations` →
  `Computing refusal directions` → `Directions ready`.
- `src/abliterate/pipeline.py:170` — `optimize_and_apply(..., on_progress=None)`
  maps the whole pre-trial warm-up onto `[0, 1]`: prompt prep + directions
  `[0, 0.6]`, `Measuring baseline refusals` at `0.65`, reference capture
  `[0.7, 1.0]`.
- `src/abliterate/optimize.py:103` — `optimize_ablation(..., on_progress=None)`
  emits `Capturing reference distribution` → `Reference distribution captured`.

### UI polish
- `src/abliterate/ui.py:46` — global progress bands (`_LOAD_BAND` … `_AFTER_BAND`)
  and `_lerp` (`:53`).
- `src/abliterate/ui.py:67` — `Job.indeterminate` flag, surfaced in
  `snapshot()` (`:94`).
- `src/abliterate/ui.py:163` — worker `set_phase` helper; `on_load` (`:171`) and
  `on_warmup` (`:176`) map stage fractions onto the global bar.
- `src/abliterate/ui.py:193` — the "before" demo step now says it can take
  several minutes on a CPU-offloaded model and shows the spinner.
- `src/abliterate/ui.py:215` — the "after" demo step also shows the spinner.
- `src/abliterate/ui.py:263` — `_model_input_error` (Ollama `name:tag` / malformed
  ids → friendly message); `:280` `_friendly_error` translates HF validation
  errors; `:288` `_start_job` validates before spawning the worker.
- `src/abliterate/ui.py` HTML/CSS/JS — `.spinner` + `@keyframes spin`,
  `.bar.indet` + `@keyframes indet`, spinner element in the progress card, and
  `tick()` toggles spinner/indeterminate mode (`:630`).

## Duplicate-load before/after evidence

**Before** (from the addendum's real run — 5 load+log events, 2 files):
```
14:07:34 loaded 243375 prompts from ...\data\harmful.txt   <- ui.py demo sample
14:13:10 loaded 243375 prompts from ...\data\harmful.txt   <- pipeline direction pass
14:13:11 loaded 327230 prompts from ...\data\harmless.txt  <- pipeline direction pass
14:13:11 loaded 243375 prompts from ...\data\harmful.txt   <- pipeline eval pass
14:13:11 loaded 327230 prompts from ...\data\harmless.txt  <- pipeline eval pass
```

**After** (3 calls on the same source, one process):
```
$ python -c "..."   # load_prompts called 3x on tmp/ui-feedback/sample.txt
INFO abliterate: loaded 3 prompts from tmp/ui-feedback/sample.txt
```
Exactly one line for three calls. The same source is now read once per process,
so the real run collapses to one line per distinct file (2 total).

## New progress phases (UI)

| Stage | Phases emitted | Bar mode |
|---|---|---|
| Model load | Resolving tokenizer, Loading tokenizer, Downloading weights / Reading weights from disk, Loading into memory, Merging LoRA adapter, Resolving architecture, Model ready | determinate, spinner during download |
| Before demo | Checking how the original model responds (…several minutes…) | spinner |
| Warm-up | Preparing prompts, Collecting harmful activations, Collecting harmless activations, Computing refusal directions, Directions ready, Measuring baseline refusals, Capturing reference distribution, Reference distribution captured | determinate |
| Trials | Searching for the best settings — trial N of M | determinate |
| After demo | Checking the abliterated model | spinner |

The bar advances continuously through the previously-flat `0.15 → 0.87` window
and never sits static: any step without a measurable fraction flips to the
spinner.

## Tests

`tests/test_data_cache.py`
- `test_load_prompts_reads_file_once` — monkeypatches `_read_text_file` and
  asserts **one** read and **one** `loaded` log record across three calls.
- `test_load_prompts_returns_independent_copies` — mutating a result does not
  poison the cache.
- `test_load_prompts_distinct_sources_are_cached_separately`.

`tests/test_ui_progress.py`
- `test_worker_emits_many_distinct_phases` — mocks model load / generation /
  optimization and runs `_run_job`; asserts **> 3 distinct phases**, that the
  indeterminate spinner was engaged, and that progress ends at `1.0`.
- `test_ollama_style_model_id_yields_friendly_error` — `Qwen3:14B` → friendly
  message, no raw HF exception.
- `test_model_input_error_accepts_hf_id_and_local_path`.

## Commands run

```
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q                       # 6 passed
.venv/bin/python -c "import abliterate.ui"          # OK
.venv/bin/python -m abliterate --help               # exit 0
.venv/bin/python -m abliterate ui --help            # exit 0
.venv/bin/python -m abliterate optimize --help      # exit 0
.venv/bin/python -m compileall -q src/abliterate     # OK
```

## Limitations / notes

- **No new dependencies.** `progress.py` is stdlib-only; `ui.py` remains
  stdlib-only; `data.py` gained no deps.
- **Signatures preserved.** All new parameters are keyword-only with defaults, so
  `cli.py`, `pipeline.py` and `ui.py` call sites are unchanged. CLI prints are
  untouched.
- **Download progress is phase-level, not byte-level.** `transformers` does not
  expose a per-shard callback through `from_pretrained`, so the download step is
  reported as an indeterminate spinner rather than a percentage. This satisfies
  the "never looks frozen" requirement; a byte-accurate bar would require
  hooking `huggingface_hub`'s tqdm (out of scope, and would add coupling).
- **Model-id validation is heuristic.** It accepts existing local paths and
  `namespace/name` ids; anything else (including `name:tag`) is rejected with
  guidance. A syntactically valid but non-existent repo id still fails at load
  time, where `_friendly_error` translates the known HF validation messages.
- **`_clear_prompt_cache` is private** and exists only for test isolation; the
  cache is process-global by design (matches "once per process" in the brief).
- The `brief-*.md` and `ADDENDUM-*.md` files were left untracked (task inputs,
  not deliverables).
