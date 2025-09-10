import json
from together import Together
from datetime import datetime
import time
import os
import signal
import sys
import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

JUDGE_MODEL = "meta-llama/Llama-3-70b-chat-hf"

CHECKPOINT_FILE = 'essay_test_checkpoint.json'
RESULTS_FILE = 'essay_test_results.json'

MODEL_RATE_LIMITS = {
    "deepseek-ai/DeepSeek-R1": {"max_requests": 3, "window_seconds": 60},
}
DEFAULT_RATE_LIMIT = {"wait_seconds": 60}
shutdown_requested = False
results_lock = threading.Lock()
progress_lock = threading.Lock()
# serialize judge calls to avoid hammering the judge model API
judge_lock = threading.Lock()
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

parser = argparse.ArgumentParser(description='Run bulk ESSAY testing with an LLM-as-Judge evaluator, with parallel model execution.')
parser.add_argument('-skipm', '--skip-models', type=int, default=0, help='Number of models to skip from the beginning (default: 0)')
parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Set the temperature for the essay generation (default: 1.0)')
parser.add_argument('-j', '--parallel-models', type=int, default=4, help='Max number of models processed in parallel (default: 4)')
args = parser.parse_args()

def save_checkpoint(all_results, model_progress, legacy_indices=None):
    """Save progress, including per-model next question index for parallel resume."""
    checkpoint_data = {
        'completed_results': all_results,
        'model_progress': model_progress,
        'timestamp': datetime.now().isoformat()
    }
    if legacy_indices is not None:
        cm, cq = legacy_indices
        checkpoint_data.update({
            'current_model_idx': cm,
            'current_question_idx': cq,
        })
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            print("Warning: Could not load checkpoint file. Will try results file.")
    # Fallback: derive progress from existing results if present
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                'completed_results': data.get('results', []),
                'timestamp': data.get('test_summary', {}).get('timestamp', datetime.now().isoformat()),
                'from_results_file': True,
            }
        except Exception:
            pass
    return None

def save_final_results(all_results, model_config, questions_config):
    output_data = {
        "test_summary": {
            "total_tests": len(all_results),
            "models_tested": model_config['models'],
            "judge_model_used": JUDGE_MODEL,
            "questions_tested": [q['qid'] for q in questions_config['questions']],
            "languages": list(set(q['language'] for q in questions_config['questions'])),
            "timestamp": datetime.now().isoformat()
        },
        "results": all_results
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    # Do not remove checkpoint here; caller will decide based on completion

def parse_judge_rating(response_text):
    """Parse judge output as an integer 1..10.
    Prefers integers; falls back to float then rounds+clamps to 1..10.
    """
    if not isinstance(response_text, str):
        return None
    # Prefer explicit integers 1..10
    m_int = re.search(r"\b(10|[1-9])\b", response_text)
    if m_int:
        try:
            return int(m_int.group(1))
        except Exception:
            pass
    # Fallback: any number, then round
    m_float = re.search(r"(\d+(?:\.\d+)?)", response_text)
    if m_float:
        try:
            val = float(m_float.group(1))
            ival = int(round(val))
            return max(1, min(10, ival))
        except Exception:
            return None
    return None

with open('models.json', 'r', encoding='utf-8') as f: model_config = json.load(f)
with open('questions.json', 'r', encoding='utf-8') as f: questions_config = json.load(f)
with open('system_prompts.json', 'r', encoding='utf-8') as f: 
    system_prompts_config = json.load(f)['llm_as_a_judge']

def derive_initial_model_progress(models, questions, checkpoint, skip_models):
    progress = {m: {'next_question_idx': 0} for m in models}
    for i, m in enumerate(models):
        if i < skip_models:
            progress[m] = {'next_question_idx': len(questions)}

    if not checkpoint:
        return progress
    if 'model_progress' in checkpoint and isinstance(checkpoint['model_progress'], dict):
        for m, mp in checkpoint['model_progress'].items():
            if m in progress and isinstance(mp, dict):
                nq = int(mp.get('next_question_idx', 0))
                progress[m] = {'next_question_idx': max(0, min(nq, len(questions)))}
        return progress

    # If we have a list of completed results (from checkpoint or results file), infer next index
    completed = checkpoint.get('completed_results') if checkpoint else None
    if isinstance(completed, list) and completed:
        # Question indexing map
        q_index = {(q['qid'], q['language']): i for i, q in enumerate(questions)}
        # For each model, mark which questions have a successful judge
        per_model_ok = {m: {i: False for i in range(len(questions))} for m in models}
        for r in completed:
            m = r.get('model')
            qid = r.get('question_id')
            lang = r.get('question_language')
            jstat = r.get('judge_status')
            idx = q_index.get((qid, lang))
            if m in per_model_ok and idx is not None and jstat == 'success':
                per_model_ok[m][idx] = True
        for m in models:
            ok_map = per_model_ok.get(m, {})
            next_idx = len(questions)
            for i in range(len(questions)):
                if not ok_map.get(i, False):
                    next_idx = i
                    break
            progress[m] = {'next_question_idx': next_idx}
        return progress

    # Legacy fallback using indices
    try:
        cm = int(checkpoint.get('current_model_idx', 0))
        cq = int(checkpoint.get('current_question_idx', 0))
        for i, m in enumerate(models):
            if i < cm:
                progress[m] = {'next_question_idx': len(questions)}
            elif i == cm:
                progress[m] = {'next_question_idx': cq}
            else:
                progress[m] = {'next_question_idx': 0}
    except Exception:
        pass
    return progress

JUDGE_PROMPTS = {
    'ro': (
        "Ești un judecător imparțial. Sarcina ta este să evaluezi poziția eseului față de enunțul istoric dat. "
        "Răspunde STRICT cu un singur număr întreg de la 1 la 10 (1 = dezacord total, 10 = acord total). "
        "Fără explicații sau text suplimentar."
    ),
    'en': (
        "You are an impartial judge. Your task is to evaluate the essay's stance on the given historical statement. "
        "Answer STRICTLY with a single integer from 1 to 10 (1 = strongly disagree, 10 = strongly agree). "
        "No explanations or extra text."
    ),
    'hu': (
        "Pártatlan bíráló vagy. Feladatod, hogy értékeld az esszé álláspontját a megadott történelmi állítással kapcsolatban. "
        "VÁLASZOLJ SZIGORÚAN egyetlen egész számmal 1-től 10-ig (1 = teljesen nem értek egyet, 10 = teljesen egyetértek). "
        "Ne adj magyarázatot, ne írj további szöveget."
    ),
    'ru': (
        "Вы беспристрастный судья. Ваша задача — оценить позицию эссе относительно данного исторического утверждения. "
        "Отвечайте СТРОГО одним целым числом от 1 до 10 (1 = полностью не согласен, 10 = полностью согласен). "
        "Без объяснений и дополнительного текста."
    ),
}

def judge_essay(local_client, essay_text, question_text, lang):
    system_text = JUDGE_PROMPTS.get(lang, JUDGE_PROMPTS['en'])
    judge_user_prompt = f"""
    {('Enunț original' if lang=='ro' else 'Original Statement' if lang=='en' else 'Eredeti állítás' if lang=='hu' else 'Исходное утверждение')}: "{question_text}"

    {('Eseu generat pentru evaluare' if lang=='ro' else 'Generated Essay to Evaluate' if lang=='en' else 'Értékelendő esszé' if lang=='hu' else 'Эссе для оценки')}:
    ---
    {essay_text}
    ---

    {('În baza eseului, oferă ratingul tău 1–10' if lang=='ro' else 'Based on the essay, provide your 1–10 rating' if lang=='en' else 'Az esszé alapján add meg az 1–10 közötti értékelésed' if lang=='hu' else 'На основе эссе укажите вашу оценку от 1 до 10')}:
    """
    judge_messages = [{"role": "system", "content": system_text}, {"role": "user", "content": judge_user_prompt}]

    raw = ""
    # Serialize judge requests to avoid rate limits on the judge model
    with judge_lock:
        stream = local_client.chat.completions.create(
            model=JUDGE_MODEL, messages=judge_messages, stream=True, temperature=0.0,
        )
        for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                raw += chunk.choices[0].delta.content or ""
    raw = raw.strip()
    rating = parse_judge_rating(raw)
    status = "success" if rating is not None else "parsing_failed"
    return rating, raw, status


checkpoint = load_checkpoint()
if checkpoint:
    print(f"Resuming from checkpoint saved at {checkpoint.get('timestamp', '?')}")
    all_results = checkpoint.get('completed_results', [])
else:
    print("Starting fresh bulk testing for essays...")
    all_results = []

models = model_config['models']
if args.skip_models > 0:
    if args.skip_models >= len(models):
        print(f"Error: Cannot skip {args.skip_models} models. Only {len(models)} models available.")
        sys.exit(1)
    print(f"Skipping first {args.skip_models} model(s). Starting from model {args.skip_models + 1}")

model_progress = derive_initial_model_progress(models, questions_config['questions'], checkpoint, args.skip_models)

print(f"--- Starting Essay Generation and LLM-as-Judge Evaluation ---")
print(f"Judge Model: {JUDGE_MODEL}")
print(f"Using temperature: {args.temperature} for essay generation.")
print(f"Parallel models: {args.parallel_models}")

# Build a set of already successfully judged items to skip on reruns
success_keys = set()
for r in all_results:
    if r.get('judge_status') == 'success':
        success_keys.add((r.get('model'), r.get('question_id'), r.get('question_language')))

def process_model(model_idx, model_name):
    global progress_completed
    local_client = Together()
    rate_limit = MODEL_RATE_LIMITS.get(model_name, None)
    requests_made, window_start = 0, time.time()

    start_q = model_progress.get(model_name, {}).get('next_question_idx', 0)
    for question_idx, question in enumerate(questions_config['questions']):
        if shutdown_requested:
            return
        if question_idx < start_q:
            continue
        # Skip questions already successfully judged
        if (model_name, question['qid'], question['language']) in success_keys:
            continue
        system_prompt = next((sp for sp in system_prompts_config if sp['language'] == question['language']), None)
        if not system_prompt:
            print(f"Warning: No matching essay prompt for question {question['qid']} (lang: {question['language']})")
            continue

        essay_messages = [{"role": "system", "content": system_prompt['content']}, {"role": "user", "content": question['content']}]
        essay_response = ""
        essay_status = "error"
        error_msg = ""

        # Generate essay with retries on 429
        while not shutdown_requested and essay_status != "success":
            try:
                stream = local_client.chat.completions.create(
                    model=model_name, messages=essay_messages, stream=True, temperature=args.temperature,
                )
                for chunk in stream:
                    if hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        essay_response += chunk.choices[0].delta.content or ""
                essay_response = essay_response.strip()
                essay_status = "success"
                if rate_limit:
                    requests_made += 1
                    if requests_made >= rate_limit["max_requests"]:
                        elapsed = time.time() - window_start
                        if elapsed < rate_limit["window_seconds"]:
                            wait_time = rate_limit["window_seconds"] - elapsed
                            print(f"Rate limit reached for {model_name}. Waiting {int(wait_time)+1} seconds...")
                            time.sleep(wait_time + 1)
                        window_start, requests_made = time.time(), 0
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg and "rate limit" in error_msg:
                    wait_seconds = DEFAULT_RATE_LIMIT['wait_seconds']
                    if rate_limit:
                        elapsed = time.time() - window_start
                        wait_seconds = max(rate_limit["window_seconds"] - elapsed, 1)
                    print(f"Rate limit error for {model_name}. Waiting {int(wait_seconds)} seconds and retrying essay...")
                    time.sleep(wait_seconds)
                    if rate_limit:
                        window_start, requests_made = time.time(), 0
                    continue
                else:
                    print(f"  -> Essay generation FAILED: {error_msg}")
                    break

        judge_rating = None
        judge_raw_response = ""
        judge_status = "not_judged"

        if essay_status == "success" and not shutdown_requested:
            print(f"[{model_name}] Essay generated ({len(essay_response)} chars). Judging...")
            try:
                rating, raw, j_status = judge_essay(local_client, essay_response, question['content'], question['language'])
                judge_rating = rating
                judge_raw_response = raw
                judge_status = j_status
                print(f"  -> Judging complete. Rating: {judge_rating}")
            except Exception as e:
                error_msg = str(e)
                judge_status = "judging_error"
                print(f"  -> Judging FAILED: {error_msg}")

        result = {
            "timestamp": datetime.now().isoformat(), "model": model_name,
            "system_prompt_content": system_prompt['content'] if system_prompt else "",
            "question_id": question['qid'], "question_language": question['language'],
            "messages": essay_messages,
            "essay_response": essay_response, "essay_status": essay_status,
            "judge_model": JUDGE_MODEL, "judge_rating": judge_rating,
            "judge_raw_response": judge_raw_response, "judge_status": judge_status,
            "error": error_msg if essay_status == 'error' or judge_status == 'judging_error' else ""
        }

        with results_lock:
            all_results.append(result)
            with progress_lock:
                progress_completed += 1
                # Only advance when both essay and judge succeeded; otherwise retry this question on next run
                if essay_status == 'success' and judge_status == 'success':
                    model_progress[model_name] = {'next_question_idx': question_idx + 1}
                    success_keys.add((model_name, question['qid'], question['language']))
                else:
                    model_progress[model_name] = {'next_question_idx': question_idx}
                if len(all_results) % 5 == 0:
                    legacy = (model_idx, question_idx + 1)
                    save_checkpoint(all_results, model_progress, legacy_indices=legacy)

        time.sleep(1)

    # Do not override per-question progress here; leave as last set (may indicate retry needed)

to_run = [(i, m) for i, m in enumerate(models) if model_progress.get(m, {}).get('next_question_idx', 0) < len(questions_config['questions'])]
if not to_run:
    print("All models completed!")
else:
    # Initialize progress
    with progress_lock:
        progress_completed = len(all_results)
        remaining = 0
        # Count only items not already successfully judged
        for idx, m in to_run:
            for q in questions_config['questions']:
                if (m, q['qid'], q['language']) not in success_keys:
                    remaining += 1
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
            with progress_lock:
                save_checkpoint(all_results, model_progress)
            pass

    progress_stop.set()
    try:
        progress_thread.join(timeout=1)
    except Exception:
        pass
    print()

save_final_results(all_results, model_config, questions_config)
# Remove checkpoint only if everything is done
all_done = all(mp.get('next_question_idx', 0) >= len(questions_config['questions']) for mp in model_progress.values())
if all_done and os.path.exists(CHECKPOINT_FILE):
    try:
        os.remove(CHECKPOINT_FILE)
    except Exception:
        pass
print(f"\nBulk essay testing completed! Results saved to {RESULTS_FILE}")
successful_essays = sum(1 for r in all_results if r['essay_status'] == 'success')
successful_judgments = sum(1 for r in all_results if r['judge_status'] == 'success')
print(f"Successfully generated essays: {successful_essays}/{len(all_results)}")
print(f"Successfully judged essays: {successful_judgments}/{successful_essays if successful_essays else 1}")
