import json
from together import Together
from datetime import datetime
import time
import os
import signal
import sys
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_RATE_LIMITS = {
    "deepseek-ai/DeepSeek-R1": {"max_requests": 3, "window_seconds": 60},
}
DEFAULT_RATE_LIMIT = {"wait_seconds": 60}

# run_temp_1: 1
# run_temp_2: 0.4
# run_temp_3: 1
# run_temp_4: 1
# run_temp_5:
CHECKPOINT_FILE = '_test_checkpoint.json'
RESULTS_FILE = '_test_results.json'

shutdown_requested = False
results_lock = threading.Lock()
progress_lock = threading.Lock()
progress_stop = threading.Event()
progress_total = 0
progress_completed = 0
progress_start_time = None

def _fmt_eta(seconds: float) -> str:
    if seconds is None or seconds == float('inf') or seconds != seconds:
        return "--:--"
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _progress_thread():
    global progress_completed
    last_render = 0
    while not progress_stop.is_set():
        # render at ~4Hz
        now = time.time()
        if now - last_render < 0.25:
            time.sleep(0.1)
            continue
        last_render = now
        with progress_lock:
            total = progress_total
            done = progress_completed
            start = progress_start_time
        pct = (done / total * 100) if total else 100.0
        bar_len = 30
        filled = int(bar_len * (done / total)) if total else bar_len
        bar = '█' * filled + '-' * (bar_len - filled)
        elapsed = (time.time() - start) if start else 0
        rate = (done / elapsed) if elapsed > 0 else 0
        eta = ((total - done) / rate) if rate > 0 else float('inf')
        line = f"[Progress] |{bar}| {done}/{total} {pct:5.1f}% ETA {_fmt_eta(eta)}"
        try:
            sys.stdout.write('\r' + line)
            sys.stdout.flush()
        except Exception:
            pass
        time.sleep(0.25)

def signal_handler(signum, frame):
    global shutdown_requested
    print("\n\nShutdown requested. Saving progress...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run bulk testing with optional model skipping, temperature control, and parallelism')
parser.add_argument('-skipm', '--skip-models', type=int, default=0, 
                    help='Number of models to skip from the beginning (default: 0)')
parser.add_argument('-t', '--temperature', type=float, default=1,
                    help='Set the temperature for the LLM response (default: 1)')
parser.add_argument('-j', '--parallel-models', type=int, default=4,
                    help='Max number of models queried in parallel (default: 4)')
args = parser.parse_args()

def save_checkpoint(all_results, model_progress, legacy_indices=None):
    """Persist progress safely. Stores per-model progress to support parallel resume.

    model_progress: dict[str, dict] -> { model_name: { 'next_question_idx': int, 'next_prompt_idx': int } }
    legacy_indices: optional tuple (current_model_idx, current_question_idx, current_prompt_idx) for backward compat.
    """
    checkpoint_data = {
        'completed_results': all_results,
        'model_progress': model_progress,
        'timestamp': datetime.now().isoformat()
    }
    if legacy_indices is not None:
        cm, cq, cp = legacy_indices
        checkpoint_data.update({
            'current_model_idx': cm,
            'current_question_idx': cq,
            'current_prompt_idx': cp,
        })
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            print("Warning: Could not load checkpoint file. Starting from beginning.")
            return None
    return None

def save_final_results(all_results, model_config, system_prompts_config, questions_config):
    output_data = {
        "test_summary": {
            "total_tests": len(all_results),
            "models_tested": model_config['models'],
            "system_prompts_available": [sp['id'] for sp in system_prompts_config['system_prompts']],
            "questions_tested": [q['qid'] for q in questions_config['questions']],
            "languages": list(set(q['language'] for q in questions_config['questions'])),
            "timestamp": datetime.now().isoformat()
        },
        "results": all_results
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

# Load configurations
with open('models.json', 'r', encoding='utf-8') as f: model_config = json.load(f)
with open('questions.json', 'r', encoding='utf-8') as f: questions_config = json.load(f)
with open('system_prompts.json', 'r', encoding='utf-8') as f: system_prompts_config = json.load(f)

def derive_initial_model_progress(models, questions, system_prompts, checkpoint, skip_models):
    """Compute per-model start positions either from checkpoint['model_progress'] or legacy indices/completed results."""
    progress = {m: {'next_question_idx': 0, 'next_prompt_idx': 0} for m in models}

    # Respect skip_models by setting those models as fully completed (skip) for scheduling
    for i, m in enumerate(models):
        if i < skip_models:
            progress[m] = {'next_question_idx': len(questions), 'next_prompt_idx': 0}

    if not checkpoint:
        return progress

    # If new-style progress present, use it directly
    if 'model_progress' in checkpoint and isinstance(checkpoint['model_progress'], dict):
        for m, mp in checkpoint['model_progress'].items():
            if m in progress and isinstance(mp, dict):
                nq = int(mp.get('next_question_idx', 0))
                np = int(mp.get('next_prompt_idx', 0))
                progress[m] = {'next_question_idx': max(0, min(nq, len(questions))), 'next_prompt_idx': max(0, np)}
        return progress

    # Legacy: attempt to reconstruct from completed_results
    try:
        completed = checkpoint.get('completed_results', []) if isinstance(checkpoint, dict) else []
        # For each model, find the max (question_idx, prompt_idx) seen, then set next accordingly
        qid_list = [q['qid'] for q in questions]
        # Build language to prompts list for quick lookup
        prompts_by_lang = {}
        for sp in system_prompts['system_prompts']:
            prompts_by_lang.setdefault(sp['language'], []).append(sp)

        # Map question_id + lang to index
        def find_question_idx(qid, lang):
            # We stored language in results under 'question_language' and 'question_id'
            # questions is ordered by qid-language groups in file; we search sequentially
            for idx, q in enumerate(questions):
                if q.get('qid') == qid and q.get('language') == lang:
                    return idx
            return None

        for r in completed:
            m = r.get('model')
            if m not in progress:
                continue
            qid = r.get('question_id')
            lang = r.get('question_language')
            sp_id = r.get('system_prompt_id')
            qidx = find_question_idx(qid, lang)
            if qidx is None:
                continue
            # Calculate next prompt index for that question
            matching_prompts = [sp for sp in prompts_by_lang.get(lang, [])]
            try:
                pidx = next((i for i, sp in enumerate(matching_prompts) if sp.get('id') == sp_id), None)
            except Exception:
                pidx = None
            if pidx is None:
                continue
            # Next index is either within same question or move to next question
            curr = progress[m]
            if qidx > curr['next_question_idx'] or (qidx == curr['next_question_idx'] and (pidx + 1) > curr['next_prompt_idx']):
                if (pidx + 1) < len(matching_prompts):
                    progress[m] = {'next_question_idx': qidx, 'next_prompt_idx': pidx + 1}
                else:
                    progress[m] = {'next_question_idx': qidx + 1, 'next_prompt_idx': 0}
    except Exception:
        # Fallback to legacy single indices
        cm = int(checkpoint.get('current_model_idx', 0))
        cq = int(checkpoint.get('current_question_idx', 0))
        cp = int(checkpoint.get('current_prompt_idx', 0))
        for i, m in enumerate(models):
            if i < cm:
                progress[m] = {'next_question_idx': len(questions), 'next_prompt_idx': 0}
            elif i == cm:
                progress[m] = {'next_question_idx': cq, 'next_prompt_idx': cp}
            else:
                progress[m] = {'next_question_idx': 0, 'next_prompt_idx': 0}
    return progress

client = Together()

# Load checkpoint or start fresh
checkpoint = load_checkpoint()
if checkpoint:
    print(f"Resuming from checkpoint saved at {checkpoint.get('timestamp', '?')}")
    all_results = checkpoint.get('completed_results', [])
else:
    print("Starting fresh bulk testing...")
    all_results = []

models = model_config['models']

if args.skip_models > 0:
    if args.skip_models >= len(models):
        print(f"Error: Cannot skip {args.skip_models} models. Only {len(models)} models available.")
        sys.exit(1)
    print(f"Skipping first {args.skip_models} model(s). Starting from model {args.skip_models + 1}")

model_progress = derive_initial_model_progress(models, questions_config['questions'], system_prompts_config, checkpoint, args.skip_models)

# Progress accounting
prompts_by_lang = {}
for sp in system_prompts_config['system_prompts']:
    prompts_by_lang.setdefault(sp['language'], 0)
    prompts_by_lang[sp['language']] += 1

def _question_prompt_count(q):
    return prompts_by_lang.get(q['language'], 0)

def process_model(model_idx, model_name):
    global progress_completed
    local_client = Together()  # safer per-thread client
    rate_limit = MODEL_RATE_LIMITS.get(model_name, None)
    requests_made, window_start = 0, time.time()

    start_q = model_progress.get(model_name, {}).get('next_question_idx', 0)
    start_p = model_progress.get(model_name, {}).get('next_prompt_idx', 0)

    total_questions = len(questions_config['questions'])
    for question_idx, question in enumerate(questions_config['questions']):
        if shutdown_requested:
            return
        if question_idx < start_q:
            continue
        matching_prompts = [sp for sp in system_prompts_config['system_prompts'] if sp['language'] == question['language']]
        if not matching_prompts:
            print(f"Warning: No matching system prompts for question {question['qid']} (lang: {question['language']})")
            continue
        for prompt_idx, system_prompt in enumerate(matching_prompts):
            if shutdown_requested:
                return
            if question_idx == start_q and prompt_idx < start_p:
                continue

            print(f"[{model_name}] System prompt: {system_prompt['id']} | Question: {question['qid']} ({question['language']})")
            messages = [{"role": "system", "content": system_prompt['content']}, {"role": "user", "content": question['content']}]

            success = False
            while not success and not shutdown_requested:
                try:
                    stream = local_client.chat.completions.create(
                        model=model_name, messages=messages, stream=True, temperature=args.temperature,
                    )
                    full_response = ""
                    for chunk in stream:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                                content = chunk.choices[0].delta.content or ""
                                full_response += content
                    result = {
                        "timestamp": datetime.now().isoformat(), "model": model_name,
                        "system_prompt_id": system_prompt['id'], "system_prompt_content": system_prompt['content'],
                        "question_id": question['qid'], "question_language": question['language'],
                        "messages": messages, "response": full_response.strip(), "status": "success"
                    }
                    if rate_limit:
                        requests_made += 1
                        if requests_made >= rate_limit["max_requests"]:
                            elapsed = time.time() - window_start
                            if elapsed < rate_limit["window_seconds"]:
                                wait_time = rate_limit["window_seconds"] - elapsed
                                print(f"Rate limit reached for {model_name}. Waiting {int(wait_time)+1} seconds...")
                                time.sleep(wait_time + 1)
                            window_start, requests_made = time.time(), 0
                    success = True
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg and "rate limit" in error_msg:
                        wait_seconds = DEFAULT_RATE_LIMIT['wait_seconds']
                        if rate_limit:
                            elapsed = time.time() - window_start
                            wait_seconds = max(rate_limit["window_seconds"] - elapsed, 1)
                        print(f"Rate limit error for {model_name}. Waiting {int(wait_seconds)} seconds and retrying...")
                        time.sleep(wait_seconds)
                        if rate_limit:
                            window_start, requests_made = time.time(), 0
                    else:
                        result = {
                            "timestamp": datetime.now().isoformat(), "model": model_name,
                            "system_prompt_id": system_prompt['id'], "system_prompt_content": system_prompt['content'],
                            "question_id": question['qid'], "question_language": question['language'],
                            "messages": messages, "response": "", "status": "error", "error": error_msg
                        }
                        print(f"Error [{model_name}]: {error_msg}")
                        success = True

            # Append result and update shared progress/checkpoint
            with results_lock:
                all_results.append(result)
                with progress_lock:
                    progress_completed += 1
                # Update per-model progress: compute next indices for this model
                next_q = question_idx
                next_p = prompt_idx + 1
                if next_p >= len(matching_prompts):
                    next_q = question_idx + 1
                    next_p = 0
                with progress_lock:
                    model_progress[model_name] = {'next_question_idx': next_q, 'next_prompt_idx': next_p}
                if len(all_results) % 5 == 0:
                    # best-effort legacy indices: current model and its positions
                    legacy = (model_idx, question_idx, prompt_idx + 1)
                    save_checkpoint(all_results, model_progress, legacy_indices=legacy)

            time.sleep(1)

    # Model finished
    with progress_lock:
        model_progress[model_name] = {'next_question_idx': len(questions_config['questions']), 'next_prompt_idx': 0}

print(f"Starting bulk testing with {len(models)} models, {len(system_prompts_config['system_prompts'])} system prompts, and {len(questions_config['questions'])} questions...")
print(f"Using temperature: {args.temperature}")
print(f"Parallel models: {args.parallel_models}")

# Launch up to N models in parallel, skipping those marked as completed
to_run = [(i, m) for i, m in enumerate(models) if model_progress.get(m, {}).get('next_question_idx', 0) < len(questions_config['questions'])]
if not to_run:
    print("All models completed!")
    sys.exit(0)

# Initialize progress totals
with progress_lock:
    # already completed includes any prior results from checkpoint
    progress_completed = len(all_results)
    remaining = 0
    for idx, m in to_run:
        start_q = model_progress[m]['next_question_idx']
        start_p = model_progress[m]['next_prompt_idx']
        for qi, q in enumerate(questions_config['questions']):
            cnt = _question_prompt_count(q)
            if qi < start_q:
                continue
            elif qi == start_q:
                remaining += max(0, cnt - start_p)
            else:
                remaining += cnt
    progress_total = progress_completed + remaining
    progress_start_time = time.time()

progress_thread = threading.Thread(target=_progress_thread, daemon=True)
progress_thread.start()

with ThreadPoolExecutor(max_workers=max(1, min(args.parallel_models, len(to_run)))) as executor:
    futures = [executor.submit(process_model, i, m) for i, m in to_run]
    try:
        for f in as_completed(futures):
            _ = f.result()
    except KeyboardInterrupt:
        # Save current progress on interrupt
        with progress_lock:
            save_checkpoint(all_results, model_progress)
        pass

# Finalization
progress_stop.set()
try:
    progress_thread.join(timeout=1)
except Exception:
    pass
# print a newline to end the progress line cleanly
print()
save_final_results(all_results, model_config, system_prompts_config, questions_config)
print(f"\nBulk testing completed! Results saved to {RESULTS_FILE}")
successful_tests = sum(1 for result in all_results if result['status'] == 'success')
failed_tests = len(all_results) - successful_tests
print(f"Successful: {successful_tests}, Failed: {failed_tests}")
