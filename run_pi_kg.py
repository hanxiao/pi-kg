#!/usr/bin/env python3
"""kgbox orchestrator — build a canonical KG from a dataroom on a minimal Pi harness.

Same philosophy as searchbox: Pi is already a complete agent (loops, calls tools, auto-compacts).
We add exactly one thing over vanilla Pi: keep nudging until the turn budget is spent, so the
agent actually runs its extract -> embed/cluster -> canonicalize -> verify loop to completion.

The whole KG-build method lives in the SYSTEM PROMPT (prompts/kg_system.txt), present every turn,
never compacted. Tools = the embedding primitives (embed/cluster/similarity/deduplicate) served by
the reused dataroom_service.py sidecar, plus Pi built-ins (bash/grep/read/write). No app.py, no
hard-coded greedy dedup — the agent does canonicalization itself with test-time compute.

Deliverable: KG.jsonl (+ KG_STATS.md) in the work dir.

Env: LLAMA_URL, MODEL_ID, CONTEXT_WINDOW, DATAROOM (.zip|dir), TURN_BUDGET, KGBOX_TOOLS,
     EMBED_MODEL, PI_BIN.
Usage: run_kgbox.py --dataroom <zip|dir> --budget <turns> [--out <dir>] [--resume]
"""
import argparse, json, os, subprocess, sys, time, zipfile, socket, threading, urllib.request, urllib.error, shutil
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
SIDE = HERE / "server" / "dataroom_service.py"
EXT = HERE / "pi" / "extensions" / "kg-tools.ts"

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def build_system():
    tz = os.environ.get("KGBOX_TZ", "America/Los_Angeles")
    try:
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo(tz)).strftime("%A, %B %d, %Y")
    except Exception:
        today = datetime.now().strftime("%A, %B %d, %Y")
    # Prompt variant for the A/B test: PROMPT_VARIANT=steps (step-by-step workflow) | outcome
    # (result/quality-bar driven, agent figures out its own steps). Default: steps.
    variant = os.environ.get("PROMPT_VARIANT", "steps").strip().lower()
    fname = {"steps": "kg_system_steps.txt", "outcome": "kg_system_outcome.txt"}.get(variant, "kg_system_steps.txt")
    return (HERE / "prompts" / fname).read_text().replace("{today}", today)

def write_pi_config(agent_dir, llama_url):
    agent_dir.mkdir(parents=True, exist_ok=True)
    ctx = int(os.environ.get("CONTEXT_WINDOW", os.environ.get("CTX_SIZE", "131072")))
    model_id = os.environ.get("MODEL_ID", "qwen3.6")
    max_tokens = int(os.environ.get("MAX_OUTPUT_TOKENS", "8192"))
    # Thinking wiring for local Qwen3.6 on llama.cpp: llama.cpp's Jinja path honors
    # chat_template_kwargs.enable_thinking (bool). pi's "qwen-chat-template" thinkingFormat emits
    # exactly that. reasoning:true + thinkingLevelMap makes pi treat this model as reasoning-capable
    # and turn thinking ON for non-off levels. Without this, pi sends NO thinking flag at all.
    (agent_dir / "models.json").write_text(json.dumps({
        "providers": {"local": {
            "baseUrl": f"{llama_url}/v1", "api": "openai-completions",
            "apiKey": os.environ.get("LLAMA_API_KEY", "sk-local"),
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False,
                       "maxTokensField": "max_tokens", "thinkingFormat": "qwen-chat-template"},
            "models": [{"id": model_id, "contextWindow": ctx, "maxTokens": max_tokens,
                         "reasoning": True,
                         "thinkingLevelMap": {"off": None, "low": "low", "medium": "medium",
                                              "high": "high", "xhigh": "high", "max": "high"}}]}}}, indent=2))
    (agent_dir / "settings.json").write_text(json.dumps({
        "defaultProvider": "local", "defaultModel": model_id,
        "defaultThinkingLevel": os.environ.get("THINKING_LEVEL", "high"),
        "enableInstallTelemetry": False,
        "compaction": {"enabled": True,
                       "reserveTokens": int(os.environ.get("COMPACTION_RESERVE_TOKENS", "40000")),
                       "keepRecentTokens": int(os.environ.get("COMPACTION_KEEP_RECENT_TOKENS", "20000"))}}, indent=2))

def boot_sidecar(job_dir, dataroom_dir, port):
    env = dict(os.environ)
    env["DATAROOM_DIR"] = str(dataroom_dir); env["DATAROOM_PORT"] = str(port)
    env["DATAROOM_CACHE_DIR"] = str(job_dir / ".cache")
    env["WORK_DIR"] = str(dataroom_dir.parent)   # embed jsonl written here (pi cwd)
    logf = open(job_dir / "sidecar.log", "a")
    return subprocess.Popen([sys.executable, str(SIDE)], env=env, stdout=logf,
                            stderr=subprocess.STDOUT, start_new_session=True)

def wait_http(url, timeout=600):
    dl = time.time() + timeout
    while time.time() < dl:
        try:
            urllib.request.urlopen(url, timeout=3); return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(1)
    return False

def prepare_dataroom(job_dir, src):
    dr = job_dir / "dataroom"
    if dr.exists(): shutil.rmtree(dr)
    dr.mkdir(parents=True)
    src = Path(src)
    if src.is_dir():
        for p in src.rglob("*"):
            if p.is_file():
                rel = p.relative_to(src); (dr / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dr / rel)
    else:
        with zipfile.ZipFile(src) as z: z.extractall(dr)
        # strip single wrapper dir
        kids = [p for p in dr.iterdir()]
        if len(kids) == 1 and kids[0].is_dir():
            inner = kids[0]
            for p in inner.iterdir(): shutil.move(str(p), str(dr / p.name))
            inner.rmdir()
    return dr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroom", required=True)
    ap.add_argument("--budget", type=int, default=int(os.environ.get("TURN_BUDGET", "12")))
    ap.add_argument("--out", default="out/kgbox_job")
    args = ap.parse_args()

    llama = os.environ.get("LLAMA_URL", "http://34.60.82.111:8080")
    job_dir = Path(args.out).resolve(); job_dir.mkdir(parents=True, exist_ok=True)
    agent_dir = job_dir / ".pi"; work_dir = job_dir     # pi cwd = job_dir; dataroom/ inside it
    dataroom_dir = prepare_dataroom(job_dir, args.dataroom)
    n_docs = sum(1 for _ in dataroom_dir.rglob("*") if _.is_file())
    print(f"[kgbox] dataroom={dataroom_dir} ({n_docs} files) | budget={args.budget} turns | llama={llama}", flush=True)

    write_pi_config(agent_dir, llama)
    port = free_port()
    side = boot_sidecar(job_dir, dataroom_dir, port)
    if not wait_http(f"http://127.0.0.1:{port}/stats", 600):
        print("[kgbox] sidecar failed to boot", flush=True); side.terminate(); sys.exit(1)
    print(f"[kgbox] sidecar up on :{port}", flush=True)

    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["PI_SKIP_VERSION_CHECK"] = "1"
    env["DATAROOM_INDEX_URL"] = f"http://127.0.0.1:{port}"
    # KG build wants the low-level embedding primitives, not the high-level QA retrieval tools.
    env.setdefault("KGBOX_TOOLS", "embed_texts,similarity,cluster,select_diverse,rerank")
    env["SEARCHBOX_TOOLS"] = env["KGBOX_TOOLS"]   # extension reads SEARCHBOX_TOOLS gate

    cmd = [os.environ.get("PI_BIN", "pi"), "--mode", "rpc", "--no-skills",
           "--append-system-prompt", build_system(),
           "--extension", str(EXT)]
    log = open(job_dir / "pi.log", "a")
    proc = subprocess.Popen(cmd, cwd=str(work_dir), env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    def send(obj):
        proc.stdin.write(json.dumps(obj) + "\n"); proc.stdin.flush()

    turn = 0
    kg = work_dir / "KG.jsonl"
    def kg_lines():
        return sum(1 for _ in open(kg)) if kg.exists() else 0

    # Protocol (matches searchbox): send {"type":"prompt",...}; a turn ends at agent_end; nudge.
    # Thin harness (searchbox-style): the ENTIRE step-by-step workflow lives in the system prompt.
    # We only kick off once and nudge with Continue. until the turn budget is spent.
    task = ("Begin now. Follow the numbered workflow in your system prompt exactly, starting at STEP 1. "
            "Actually create/append KG.jsonl this turn - do not just plan.")
    send({"type": "prompt", "message": task})

    turn_timeout = int(os.environ.get("TURN_TIMEOUT", "900"))
    stop_reason = "budget_spent"
    try:
        while turn < args.budget:
            wd = threading.Timer(turn_timeout, lambda: send({"type": "abort"}))
            wd.start(); ended = False
            try:
                while True:
                    line = proc.stdout.readline()
                    if line == "":
                        break
                    if '"type":"message_update"' in line:
                        continue
                    log.write(line); log.flush()
                    # A turn is over when pi finishes the cycle. pi 0.81 emits agent_end on
                    # normal completion and agent_settled after an abort/compaction-settle; treat
                    # BOTH as the turn boundary so we always re-nudge and never hang.
                    if '"type":"agent_end"' in line or '"type":"agent_settled"' in line:
                        ended = True; break
            finally:
                wd.cancel()
            if not ended:
                stop_reason = "pi_exited"; break
            turn += 1
            print(f"[kgbox] turn {turn}/{args.budget} | KG.jsonl lines={kg_lines()}", flush=True)
            if turn >= args.budget:
                break
            # The ONLY thing we add over vanilla pi: keep nudging until the budget is spent.
            # No state/coverage bookkeeping in the harness - the agent tracks its own progress
            # (which docs are done) via a state file it maintains, per its system prompt.
            send({"type": "prompt", "message": "Continue."})
    except Exception as e:
        stop_reason = f"error:{e}"
    finally:
        try: send({"type": "abort"})
        except Exception: pass
        try: proc.stdin.close()
        except Exception: pass
        time.sleep(2)
        try: proc.terminate()
        except Exception: pass
        try: side.terminate()
        except Exception: pass

    stats = {"turns": turn, "stop_reason": stop_reason, "kg_lines": kg_lines(),
             "kg_path": str(kg), "kg_stats_md": str(work_dir / "KG_STATS.md")}
    json.dump(stats, open(job_dir / "run_meta.json", "w"), indent=2)
    print("\n=== KGBOX RUN DONE ===")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
