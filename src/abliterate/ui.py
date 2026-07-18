"""A friendly local web UI for abliteration.

Zero third-party dependencies: a stdlib ``http.server`` serves one self-contained
page and a tiny JSON API. Designed for someone who doesn't know what a "residual
stream" is -- paste a model, pick how thorough to be, click a button, watch it
work, and try the result. The heavy lifting reuses the same pipeline the CLI does.

Run with ``abliterate ui``.

Endpoints:
    GET  /              the single-page app
    POST /api/start     begin an abliteration job  -> {ok} | {error}
    GET  /api/status    current job state (polled by the page)
    POST /api/test      generate a reply from the finished model
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

logger = logging.getLogger("abliterate")

# Thoroughness presets: friendly labels -> optimizer/data settings.
PRESETS = {
    "quick": dict(n_samples=64, n_trials=24, n_startup_trials=8,
                  n_eval_harmful=16, n_eval_harmless=16, max_new_tokens=24),
    "balanced": dict(n_samples=128, n_trials=60, n_startup_trials=20,
                     n_eval_harmful=32, n_eval_harmless=32, max_new_tokens=32),
    "thorough": dict(n_samples=160, n_trials=150, n_startup_trials=40,
                     n_eval_harmful=48, n_eval_harmless=48, max_new_tokens=40),
}
_MAX_LOG_LINES = 400


# --------------------------------------------------------------------------- #
# Job state
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    state: str = "idle"          # idle | running | done | error
    phase: str = ""
    progress: float = 0.0        # 0..1
    log: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    result_dir: Optional[str] = None
    before_samples: list = field(default_factory=list)
    after_samples: list = field(default_factory=list)
    error: Optional[str] = None
    bundle: object = None        # kept in memory for the "try it" box
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def add_log(self, line: str) -> None:
        with self._lock:
            self.log.append(line)
            if len(self.log) > _MAX_LOG_LINES:
                del self.log[: len(self.log) - _MAX_LOG_LINES]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "phase": self.phase,
                "progress": round(self.progress, 4),
                "log": self.log[-60:],
                "metrics": self.metrics,
                "result_dir": self.result_dir,
                "before_samples": self.before_samples,
                "after_samples": self.after_samples,
                "error": self.error,
                "can_test": self.bundle is not None and self.state == "done",
            }


JOB = Job()


class _JobLogHandler(logging.Handler):
    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    def emit(self, record):
        try:
            self.job.add_log(record.getMessage())
        except Exception:  # pragma: no cover - logging must never crash the job
            pass


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _run_job(job: Job, params: dict) -> None:
    """Background worker: load -> sample 'before' -> optimize -> sample 'after'."""
    from .config import Config, DataConfig, ModelConfig, OptimizeConfig
    from .data import load_prompts, sample_prompts
    from .evaluate import generate_completions, is_refusal
    from .model_utils import load_model_and_tokenizer
    from .pipeline import optimize_and_apply

    handler = _JobLogHandler(job)
    logger.addHandler(handler)
    try:
        preset = dict(PRESETS.get(params.get("thoroughness", "balanced"), PRESETS["balanced"]))
        trials = params.get("trials")
        if trials:
            preset["n_trials"] = int(trials)
            preset["n_startup_trials"] = max(4, min(preset["n_startup_trials"], int(trials) // 3))

        model_path = (params.get("model") or "").strip()
        device_map = (params.get("device_map") or "auto").strip() or "auto"
        load_in_4bit = bool(params.get("load_in_4bit", False))
        save = "bundle" if params.get("output_format") == "bundle" else "model"
        cfg = Config(
            model=ModelConfig(
                path=model_path,
                lora_adapter=(params.get("lora") or "").strip() or None,
                dtype=params.get("dtype", "bfloat16"),
                device_map=None if device_map.lower() in ("", "none", "cpu") else device_map,
                trust_remote_code=bool(params.get("trust_remote_code", False)),
                load_in_4bit=load_in_4bit,
            ),
            data=DataConfig(n_samples=preset.pop("n_samples"), seed=0),
            optimize=OptimizeConfig(
                kl_weight=float(params.get("kl_weight") or 1.0),
                optimize_embed=bool(params.get("optimize_embed", False)),
                seed=0,
                **preset,
            ),
            output_dir=(params.get("output_dir") or "output/abliterated-model").strip(),
        )

        job.set(state="running", phase="Loading the model", progress=0.03)
        job.add_log(f"Loading {model_path} ...")
        bundle = load_model_and_tokenizer(cfg.model)
        job.set(bundle=bundle)

        demo = sample_prompts(
            load_prompts(cfg.data.harmful, default_file="harmful.txt"), 3, seed=7
        )
        job.set(phase="Checking how the original model responds", progress=0.09)
        before = generate_completions(bundle, demo, max_new_tokens=48, batch_size=len(demo))
        job.set(before_samples=[
            {"prompt": p, "response": _clip(c), "refused": is_refusal(c)}
            for p, c in zip(demo, before)
        ])

        def on_trial(done: int, total: int) -> None:
            job.set(phase=f"Searching for the best settings — trial {done} of {total}",
                    progress=0.15 + 0.72 * (done / max(total, 1)))

        job.set(phase="Searching for the best settings", progress=0.15)
        out = optimize_and_apply(cfg, bundle=bundle, on_trial=on_trial, save=save)

        # The in-memory model behaves abliterated either way (baked weights or
        # attached hooks), so 'after' samples and the try-it box work.
        job.set(phase="Checking the abliterated model", progress=0.9)
        after = generate_completions(bundle, demo, max_new_tokens=48, batch_size=len(demo))
        job.set(after_samples=[
            {"prompt": p, "response": _clip(c), "refused": is_refusal(c)}
            for p, c in zip(demo, after)
        ])

        job.set(
            metrics=_read_metrics(out),
            result_dir=str(out),
            phase="Done",
            progress=1.0,
            state="done",
        )
        job.add_log(f"Saved to {out}")
    except Exception as exc:  # surface a friendly error, keep the server alive
        logger.error("job failed: %s", exc)
        job.add_log(traceback.format_exc().strip().splitlines()[-1])
        job.set(state="error", error=str(exc), phase="Failed")
    finally:
        logger.removeHandler(handler)


def _clip(text: str, n: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n] + "…"


def _read_metrics(out) -> dict:
    """Metrics live in the model manifest or the bundle's json, depending on mode."""
    out = Path(out)
    for name in ("abliteration_manifest.json", "bundle.json"):
        f = out / name
        if f.exists():
            return json.loads(f.read_text()).get("metrics", {})
    return {}


def _start_job(params: dict) -> tuple[bool, str]:
    if JOB.state == "running":
        return False, "A job is already running. Please wait for it to finish."
    if not (params.get("model") or "").strip():
        return False, "Please enter a model name or path."
    # Reset state for a fresh run.
    JOB.set(state="running", phase="Starting…", progress=0.0, log=[], metrics={},
            result_dir=None, before_samples=[], after_samples=[], error=None, bundle=None)
    threading.Thread(target=_run_job, args=(JOB, params), daemon=True).start()
    return True, "started"


def _test_prompt(prompt: str) -> dict:
    from .evaluate import generate_completions, is_refusal

    if JOB.bundle is None or JOB.state != "done":
        return {"error": "Finish an abliteration first, then you can test the model."}
    completion = generate_completions(
        JOB.bundle, [prompt], max_new_tokens=200, batch_size=1
    )[0]
    return {"response": completion.strip(), "refused": is_refusal(completion)}


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console clean
        logger.debug("ui: " + (args[0] % args[1:] if args else ""))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send_json(JOB.snapshot())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/start":
            ok, msg = _start_job(self._read_json())
            self._send_json({"ok": ok, "message": msg}, 200 if ok else 409)
        elif self.path == "/api/test":
            prompt = (self._read_json().get("prompt") or "").strip()
            if not prompt:
                self._send_json({"error": "Enter a prompt."}, 400)
            else:
                self._send_json(_test_prompt(prompt))
        else:
            self._send_json({"error": "not found"}, 404)


def launch(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    """Start the web UI server (blocks until Ctrl-C)."""
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    logger.info("abliterate UI running at %s  (Ctrl-C to stop)", url)
    print(f"\n  Abliterate UI is live:  {url}\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


# --------------------------------------------------------------------------- #
# The page (self-contained: no external assets, works offline)
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Abliterate</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#ffffff; --ink:#1a1c1f; --muted:#5f6672; --line:#e5e8ec;
    --accent:#5b6cff; --accent-ink:#ffffff; --ok:#1a9d63; --bad:#d1483b; --soft:#eef0ff;
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#0f1115; --card:#171a21; --ink:#e8eaed; --muted:#9aa2af; --line:#262b34;
      --accent:#7b8bff; --accent-ink:#0f1115; --ok:#4cc38a; --bad:#f0776b; --soft:#1b2030; }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:760px;margin:0 auto;padding:28px 20px 80px}
  header h1{font-size:26px;margin:0 0 4px}
  header p{color:var(--muted);margin:0 0 22px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:20px;margin:0 0 18px}
  .card h2{font-size:15px;margin:0 0 12px;letter-spacing:.02em}
  .step{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;
    border-radius:50%;background:var(--soft);color:var(--accent);font-size:12px;font-weight:700;margin-right:8px}
  label{display:block;font-weight:600;margin:0 0 6px;font-size:13px}
  .hint{color:var(--muted);font-size:12.5px;margin:4px 0 0}
  input[type=text],input[type=number],select,textarea{width:100%;padding:10px 12px;
    border:1px solid var(--line);border-radius:9px;background:var(--bg);color:var(--ink);
    font:inherit;outline:none}
  input:focus,select:focus,textarea:focus{border-color:var(--accent)}
  .row{margin:0 0 14px}
  .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
  @media (max-width:560px){.grid3{grid-template-columns:1fr}}
  .opt{border:1.5px solid var(--line);border-radius:11px;padding:12px;cursor:pointer;background:var(--bg)}
  .opt.sel{border-color:var(--accent);background:var(--soft)}
  .opt b{display:block;margin-bottom:2px}
  .opt small{color:var(--muted)}
  details{margin-top:6px}
  details summary{cursor:pointer;color:var(--accent);font-weight:600;font-size:13px}
  .adv{margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media (max-width:560px){.adv{grid-template-columns:1fr}}
  button.primary{width:100%;padding:13px;border:0;border-radius:11px;background:var(--accent);
    color:var(--accent-ink);font-weight:700;font-size:15px;cursor:pointer}
  button.primary:disabled{opacity:.5;cursor:default}
  button.ghost{padding:10px 14px;border:1px solid var(--line);border-radius:9px;background:var(--bg);
    color:var(--ink);font-weight:600;cursor:pointer}
  .hidden{display:none}
  .bar{height:9px;background:var(--soft);border-radius:6px;overflow:hidden;margin:10px 0}
  .bar > i{display:block;height:100%;width:0;background:var(--accent);transition:width .4s ease}
  .phase{font-weight:600}
  pre.log{background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:10px;
    max-height:180px;overflow:auto;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
    color:var(--muted);white-space:pre-wrap;margin:10px 0 0}
  .metrics{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:0 0 14px}
  @media (max-width:560px){.metrics{grid-template-columns:1fr}}
  .metric{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px;text-align:center}
  .metric .n{font-size:22px;font-weight:800}
  .metric .l{font-size:12px;color:var(--muted)}
  .ex{border:1px solid var(--line);border-radius:10px;padding:12px;margin:0 0 10px;background:var(--bg)}
  .ex .q{font-weight:600;margin-bottom:8px}
  .ex .cols{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  @media (max-width:560px){.ex .cols{grid-template-columns:1fr}}
  .ex .cols > div{font-size:13px}
  .tag{display:inline-block;font-size:11px;font-weight:700;padding:1px 7px;border-radius:20px;margin-left:6px}
  .tag.ref{background:rgba(209,72,59,.15);color:var(--bad)}
  .tag.ok{background:rgba(26,157,99,.15);color:var(--ok)}
  .banner{border-radius:10px;padding:12px 14px;margin:0 0 16px;font-size:14px}
  .banner.err{background:rgba(209,72,59,.12);color:var(--bad);border:1px solid var(--bad)}
  .resp{background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:10px;
    white-space:pre-wrap;min-height:40px;margin-top:10px;font-size:13.5px}
  .foot{color:var(--muted);font-size:12px;text-align:center;margin-top:26px}
  code{background:var(--soft);padding:1px 5px;border-radius:5px;font-size:12.5px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Abliterate</h1>
    <p>Remove refusals from a language model. Everything runs on your own machine.</p>
  </header>

  <div id="err" class="banner err hidden"></div>

  <!-- setup -->
  <div id="setup">
    <div class="card">
      <h2><span class="step">1</span>Your model</h2>
      <div class="row">
        <label>Model name or folder</label>
        <input id="model" type="text" placeholder="e.g. Qwen/Qwen2.5-1.5B-Instruct  or  /path/to/model">
        <p class="hint">A Hugging Face model ID, or a folder on this computer.</p>
      </div>
      <div class="row">
        <label>LoRA adapter <span style="font-weight:400;color:var(--muted)">(optional)</span></label>
        <input id="lora" type="text" placeholder="folder with a LoRA / reforge adapter to merge first">
      </div>
    </div>

    <div class="card">
      <h2><span class="step">2</span>How thorough should it be?</h2>
      <div class="grid3" id="modes">
        <div class="opt" data-mode="quick"><b>Quick</b><small>fewest tries — a fast first pass</small></div>
        <div class="opt sel" data-mode="balanced"><b>Balanced</b><small>good results, reasonable time</small></div>
        <div class="opt" data-mode="thorough"><b>Thorough</b><small>most tries — best quality, slowest</small></div>
      </div>
      <p class="hint">It automatically tries many settings and keeps the one that removes the most
        refusals while changing the model the least.</p>

      <details>
        <summary>Advanced settings</summary>
        <div class="adv">
          <div>
            <label>Save to folder</label>
            <input id="output" type="text" value="output/abliterated-model">
          </div>
          <div>
            <label>Precision</label>
            <select id="dtype">
              <option value="bfloat16">bfloat16 (recommended)</option>
              <option value="float16">float16</option>
              <option value="float32">float32 (CPU / most compatible)</option>
            </select>
          </div>
          <div>
            <label>Output</label>
            <select id="output_format">
              <option value="model">Full model folder</option>
              <option value="bundle">Lightweight bundle (tiny sidecar)</option>
            </select>
            <p class="hint">A bundle is a few KB applied on load — needed for 4-bit models.</p>
          </div>
          <div>
            <label style="display:flex;align-items:center;gap:8px;font-weight:600">
              <input id="lowvram" type="checkbox" style="width:auto"> Low VRAM (4-bit)
            </label>
            <p class="hint">Load big models quantized (GPU only). Saves a bundle.</p>
          </div>
          <div>
            <label>Protect abilities (KL weight)</label>
            <input id="kl" type="number" step="0.1" min="0" value="1.0">
            <p class="hint">Higher = keep the model's skills more intact.</p>
          </div>
          <div>
            <label>Tries (override)</label>
            <input id="trials" type="number" min="0" placeholder="leave blank to use preset">
          </div>
          <div>
            <label style="display:flex;align-items:center;gap:8px;font-weight:600">
              <input id="trust" type="checkbox" style="width:auto"> Trust remote code
            </label>
            <p class="hint">Only for models that require it.</p>
          </div>
        </div>
      </details>
    </div>

    <button id="go" class="primary">Remove refusals</button>
  </div>

  <!-- progress -->
  <div id="progress" class="card hidden">
    <h2>Working…</h2>
    <div class="phase" id="phase">Starting…</div>
    <div class="bar"><i id="fill"></i></div>
    <details><summary>Show details</summary><pre class="log" id="log"></pre></details>
  </div>

  <!-- results -->
  <div id="results" class="hidden">
    <div class="card">
      <h2>Result</h2>
      <div class="metrics">
        <div class="metric"><div class="n" id="m-before">–</div><div class="l">refusals before</div></div>
        <div class="metric"><div class="n" id="m-after">–</div><div class="l">refusals after</div></div>
        <div class="metric"><div class="n" id="m-kl">–</div><div class="l">change to model (KL)</div></div>
      </div>
      <p class="hint">Saved to <code id="savedpath">–</code></p>
      <div id="examples"></div>
      <button id="again" class="ghost" style="margin-top:8px">Start another</button>
    </div>

    <div class="card">
      <h2>Try it</h2>
      <textarea id="tprompt" rows="3" placeholder="Ask the abliterated model something…"></textarea>
      <div style="margin-top:10px"><button id="send" class="ghost">Send</button></div>
      <div class="resp hidden" id="resp"></div>
    </div>
  </div>

  <div class="foot">Runs locally · use only on models you're allowed to modify</div>
</div>

<script>
let mode = "balanced", polling = null;

document.querySelectorAll("#modes .opt").forEach(el => {
  el.onclick = () => {
    document.querySelectorAll("#modes .opt").forEach(o => o.classList.remove("sel"));
    el.classList.add("sel"); mode = el.dataset.mode;
  };
});

function showErr(msg){ const e=document.getElementById("err"); e.textContent=msg; e.classList.remove("hidden"); }
function hideErr(){ document.getElementById("err").classList.add("hidden"); }

document.getElementById("go").onclick = async () => {
  hideErr();
  const body = {
    model: document.getElementById("model").value,
    lora: document.getElementById("lora").value,
    thoroughness: mode,
    output_dir: document.getElementById("output").value,
    dtype: document.getElementById("dtype").value,
    kl_weight: document.getElementById("kl").value,
    trials: document.getElementById("trials").value,
    trust_remote_code: document.getElementById("trust").checked,
    output_format: document.getElementById("output_format").value,
    load_in_4bit: document.getElementById("lowvram").checked,
  };
  if(body.load_in_4bit){ body.output_format = "bundle"; }
  if(!body.model.trim()){ showErr("Please enter a model name or path."); return; }
  const r = await fetch("/api/start", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  const j = await r.json();
  if(!j.ok){ showErr(j.message || "Could not start."); return; }
  document.getElementById("setup").classList.add("hidden");
  document.getElementById("results").classList.add("hidden");
  document.getElementById("progress").classList.remove("hidden");
  document.getElementById("go").disabled = true;
  poll();
};

function poll(){ if(polling) clearInterval(polling); polling = setInterval(tick, 1000); tick(); }

async function tick(){
  let s; try { s = await (await fetch("/api/status")).json(); } catch(e){ return; }
  document.getElementById("phase").textContent = s.phase || "";
  document.getElementById("fill").style.width = Math.round((s.progress||0)*100) + "%";
  document.getElementById("log").textContent = (s.log||[]).join("\n");
  document.getElementById("log").scrollTop = 1e9;
  if(s.state === "done"){ clearInterval(polling); renderResults(s); }
  else if(s.state === "error"){ clearInterval(polling); showErr(s.error||"Something went wrong.");
    document.getElementById("progress").classList.add("hidden");
    document.getElementById("setup").classList.remove("hidden");
    document.getElementById("go").disabled = false; }
}

function pct(x){ return (x==null) ? "–" : Math.round(x*100) + "%"; }

function renderResults(s){
  document.getElementById("progress").classList.add("hidden");
  document.getElementById("results").classList.remove("hidden");
  const m = s.metrics || {};
  document.getElementById("m-before").textContent = pct(m.refusal_rate_before);
  document.getElementById("m-after").textContent  = pct(m.refusal_rate_after);
  document.getElementById("m-kl").textContent     = (m.kl_divergence==null?"–":(+m.kl_divergence).toFixed(2));
  document.getElementById("savedpath").textContent = s.result_dir || "–";

  const ex = document.getElementById("examples"); ex.innerHTML = "";
  const before = s.before_samples||[], after = s.after_samples||[];
  for(let i=0;i<after.length;i++){
    const b = before[i]||{}, a = after[i]||{};
    const div = document.createElement("div"); div.className="ex";
    div.innerHTML = `<div class="q">${esc(a.prompt||"")}</div>
      <div class="cols">
        <div><b>Before</b> ${tag(b.refused)}<div>${esc(b.response||"")}</div></div>
        <div><b>After</b> ${tag(a.refused)}<div>${esc(a.response||"")}</div></div>
      </div>`;
    ex.appendChild(div);
  }
}
function tag(ref){ return ref===undefined?"" : (ref?'<span class="tag ref">refused</span>':'<span class="tag ok">answered</span>'); }
function esc(t){ return (t||"").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

document.getElementById("again").onclick = () => {
  document.getElementById("results").classList.add("hidden");
  document.getElementById("setup").classList.remove("hidden");
  document.getElementById("go").disabled = false; hideErr();
};

document.getElementById("send").onclick = async () => {
  const p = document.getElementById("tprompt").value.trim();
  const box = document.getElementById("resp");
  if(!p){ return; }
  box.classList.remove("hidden"); box.textContent = "Thinking…";
  const r = await fetch("/api/test", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({prompt:p})});
  const j = await r.json();
  box.textContent = j.error ? j.error : (j.response || "(no output)");
};
</script>
</body>
</html>
"""
