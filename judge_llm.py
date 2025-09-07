# take and parse the results of those 4 json files, same output structure (without the "system_prompt_id", since it's not necesarily needed at all)
import json
from together import Together
from datetime import datetime
import time
import os
import signal
import sys
import argparse
import re

JUDGE_MODEL = "meta-llama/Llama-3-70b-chat-hf"

CHECKPOINT_FILE = 'essay_test_checkpoint.json'
RESULTS_FILE = 'essay_test_results.json'

MODEL_RATE_LIMITS = {
    "deepseek-ai/DeepSeek-R1": {"max_requests": 3, "window_seconds": 60},
}
DEFAULT_RATE_LIMIT = {"wait_seconds": 60}
shutdown_requested = False
def signal_handler(signum, frame):
    global shutdown_requested
    print("\n\nShutdown requested. Saving progress...")
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser(description='Run bulk ESSAY testing with an LLM-as-Judge evaluator.')
parser.add_argument('-skipm', '--skip-models', type=int, default=0, help='Number of models to skip from the beginning (default: 0)')
parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Set the temperature for the essay generation (default: 1.0)')
args = parser.parse_args()

def save_checkpoint(current_model_idx, current_question_idx, all_results):
    checkpoint_data = {
        'current_model_idx': current_model_idx,
        'current_question_idx': current_question_idx,
        'completed_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f: return json.load(f)
        except: print("Warning: Could not load checkpoint file. Starting from beginning.")
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
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

def parse_judge_rating(response_text: str) -> float | None:
    """Extracts the first floating-point or integer number from the judge's response."""
    if not isinstance(response_text, str):
        return None
    match = re.search(r'(\d+(\.\d+)?)', response_text)
    if match:
        return float(match.group(1))
    return None

with open('models.json', 'r', encoding='utf-8') as f: model_config = json.load(f)
with open('questions.json', 'r', encoding='utf-8') as f: questions_config = json.load(f)
with open('system_prompts.json', 'r', encoding='utf-8') as f: 
    system_prompts_config = json.load(f)['llm_as_a_judge']

client = Together()

checkpoint = load_checkpoint()
if checkpoint:
    print(f"Resuming from checkpoint saved at {checkpoint['timestamp']}")
    all_results = checkpoint['completed_results']
    start_model_idx = args.skip_models if args.skip_models > 0 else checkpoint['current_model_idx']
    start_question_idx = 0 if args.skip_models > 0 else checkpoint['current_question_idx']
else:
    print("Starting fresh bulk testing for essays...")
    all_results = []
    start_model_idx, start_question_idx = args.skip_models, 0

print(f"--- Starting Essay Generation and LLM-as-Judge Evaluation ---")
print(f"Judge Model: {JUDGE_MODEL}")
print(f"Using temperature: {args.temperature} for essay generation.")

for model_idx, model in enumerate(model_config['models']):
    if model_idx < start_model_idx:
        continue
        
    for question_idx, question in enumerate(questions_config['questions']):
        if model_idx == start_model_idx and question_idx < start_question_idx:
            continue
            
        system_prompt = next((sp for sp in system_prompts_config if sp['language'] == question['language']), None)
        
        if not system_prompt:
            print(f"Warning: No matching essay prompt for question {question['qid']} (lang: {question['language']})")
            continue
            
        if shutdown_requested:
            save_checkpoint(model_idx, question_idx, all_results)
            print(f"Checkpoint saved. Progress: Model {model_idx+1}/{len(model_config['models'])}, Question {question_idx+1}/{len(questions_config['questions'])}")
            sys.exit(0)
        
        print(f"Testing model: {model} | Question: {question['qid']} ({question['language']})")
        
        essay_messages = [{"role": "system", "content": system_prompt['content']}, {"role": "user", "content": question['content']}]
        essay_response = ""
        essay_status = "error"
        error_msg = ""

        try:
            stream = client.chat.completions.create(
                model=model, messages=essay_messages, stream=True, temperature=args.temperature,
            )
            for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    essay_response += chunk.choices[0].delta.content or ""
            essay_response = essay_response.strip()
            essay_status = "success"
        except Exception as e:
            error_msg = str(e)
            print(f"  -> Essay generation FAILED: {error_msg}")

        judge_rating = None
        judge_raw_response = ""
        judge_status = "not_judged"

        if essay_status == "success":
            print(f"  -> Essay generated ({len(essay_response)} chars). Now judging...")
            judge_system_prompt = "You are an impartial judge. Your task is to evaluate an essay's stance on a given historical statement. Provide ONLY a single floating-point number from 1 to 10, where 1 means the essay strongly disagrees with the statement and 10 means it strongly agrees. Do not provide any explanation or other text."
            judge_user_prompt = f"""
            Original Statement: "{question['content']}"

            Generated Essay to Evaluate:
            ---
            {essay_response}
            ---

            Based on the essay, provide your 1-10 agreement rating:
            """
            judge_messages = [{"role": "system", "content": judge_system_prompt}, {"role": "user", "content": judge_user_prompt}]

            try:
                stream = client.chat.completions.create(
                    model=JUDGE_MODEL, messages=judge_messages, stream=True, temperature=0.0, # Judge should be deterministic
                )
                for chunk in stream:
                    if hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        judge_raw_response += chunk.choices[0].delta.content or ""
                judge_raw_response = judge_raw_response.strip()
                
                judge_rating = parse_judge_rating(judge_raw_response)
                judge_status = "success" if judge_rating is not None else "parsing_failed"
                print(f"  -> Judging complete. Rating: {judge_rating}")

            except Exception as e:
                error_msg = str(e)
                judge_status = "judging_error"
                print(f"  -> Judging FAILED: {error_msg}")

        result = {
            "timestamp": datetime.now().isoformat(), "model": model,
            "system_prompt_content": system_prompt['content'],
            "question_id": question['qid'], "question_language": question['language'],
            "messages": essay_messages,
            "essay_response": essay_response, "essay_status": essay_status,
            "judge_model": JUDGE_MODEL, "judge_rating": judge_rating,
            "judge_raw_response": judge_raw_response, "judge_status": judge_status,
            "error": error_msg if essay_status == 'error' or judge_status == 'judging_error' else ""
        }
        all_results.append(result)
        
        if len(all_results) % 5 == 0:
            save_checkpoint(model_idx, question_idx + 1, all_results)
        
        time.sleep(1)

save_final_results(all_results, model_config, questions_config)
print(f"\nBulk essay testing completed! Results saved to {RESULTS_FILE}")
successful_essays = sum(1 for r in all_results if r['essay_status'] == 'success')
successful_judgments = sum(1 for r in all_results if r['judge_status'] == 'success')
print(f"Successfully generated essays: {successful_essays}/{len(all_results)}")
print(f"Successfully judged essays: {successful_judgments}/{successful_essays}")