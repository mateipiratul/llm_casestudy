import json
from together import Together
from datetime import datetime
import time
import os
import signal
import sys
import argparse

# --- Main Configuration ---
# Define the single model to be tested
TARGET_MODEL = "author/your_model-name"

# Use unique filenames for this specific model's test run
CHECKPOINT_FILE = '_test_checkpoint.json'
RESULTS_FILE = '_test_results.json'

# --- Rate Limiting (Optional) ---
# You can add a specific rate limit for Gemma if needed
MODEL_RATE_LIMITS = {
    # "google/gemma-2-27b-it": {"max_requests": 5, "window_seconds": 60},
}
DEFAULT_RATE_LIMIT = {"wait_seconds": 60}

# --- Graceful Shutdown ---
shutdown_requested = False
def signal_handler(signum, frame):
    global shutdown_requested
    print("\n\nShutdown requested. Saving progress...")
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# --- Command Line Arguments ---
# The --skip-models argument has been removed as it is not needed
parser = argparse.ArgumentParser(description='Run bulk testing for a single model with temperature control')
parser.add_argument('-t', '--temperature', type=float, default=1.0,
                    help='Set the temperature for the LLM response (default: 1.0)')
args = parser.parse_args()

# --- Helper Functions (Unchanged) ---
def save_checkpoint(current_question_idx, current_prompt_idx, all_results):
    checkpoint_data = {
        'current_question_idx': current_question_idx,
        'current_prompt_idx': current_prompt_idx,
        'completed_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except:
            print("Warning: Could not load checkpoint file. Starting from beginning.")
            return None
    return None

def save_final_results(all_results, system_prompts_config, questions_config):
    output_data = {
        "test_summary": {
            "total_tests": len(all_results),
            "models_tested": [TARGET_MODEL], # Hard-code the model name
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

# --- Main Script Logic ---

# Load configurations (models.json is no longer needed)
with open('questions.json', 'r', encoding='utf-8') as f: questions_config = json.load(f)
with open('system_prompts.json', 'r', encoding='utf-8') as f: system_prompts_config = json.load(f)

client = Together()

# Load checkpoint or start fresh
checkpoint = load_checkpoint()
if checkpoint:
    print(f"Resuming from checkpoint saved at {checkpoint['timestamp']}")
    all_results = checkpoint['completed_results']
    start_question_idx = checkpoint['current_question_idx']
    start_prompt_idx = checkpoint['current_prompt_idx']
else:
    print("Starting fresh test run...")
    all_results = []
    start_question_idx = 0
    start_prompt_idx = 0

print(f"--- Starting Test for Model: {TARGET_MODEL} ---")
print(f"Testing with {len(system_prompts_config['system_prompts'])} system prompts and {len(questions_config['questions'])} questions...")
print(f"Using temperature: {args.temperature}")

# Rate limit tracking for the model
rate_limit = MODEL_RATE_LIMITS.get(TARGET_MODEL, None)
requests_made, window_start = 0, time.time()

# Main testing loop (simplified: no longer loops over models)
for question_idx, question in enumerate(questions_config['questions']):
    if question_idx < start_question_idx:
        continue
        
    matching_prompts = [sp for sp in system_prompts_config['system_prompts'] if sp['language'] == question['language']]
    
    if not matching_prompts:
        print(f"Warning: No matching system prompts for question {question['qid']} (lang: {question['language']})")
        continue
        
    for prompt_idx, system_prompt in enumerate(matching_prompts):
        if question_idx == start_question_idx and prompt_idx < start_prompt_idx:
            continue
            
        if shutdown_requested:
            save_checkpoint(question_idx, prompt_idx, all_results)
            print(f"Checkpoint saved. Progress: Question {question_idx+1}/{len(questions_config['questions'])}, Prompt {prompt_idx+1}/{len(matching_prompts)}")
            sys.exit(0)
        
        print(f"System prompt: {system_prompt['id']} | Question: {question['qid']} ({question['language']})")
        
        messages = [{"role": "system", "content": system_prompt['content']}, {"role": "user", "content": question['content']}]
        
        success = False
        while not success:
            try:
                stream = client.chat.completions.create(
                    model=TARGET_MODEL, # Use the hard-coded model name
                    messages=messages,
                    stream=True,
                    temperature=args.temperature,
                )
                full_response = ""
                for chunk in stream:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content or ""
                            full_response += content
                result = {
                    "timestamp": datetime.now().isoformat(), "model": TARGET_MODEL,
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
                            print(f"Rate limit reached. Waiting {int(wait_time)+1} seconds...")
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
                    print(f"Rate limit error. Waiting {int(wait_seconds)} seconds and retrying...")
                    time.sleep(wait_seconds)
                    if rate_limit: window_start, requests_made = time.time(), 0
                else:
                    result = {
                        "timestamp": datetime.now().isoformat(), "model": TARGET_MODEL,
                        "system_prompt_id": system_prompt['id'], "system_prompt_content": system_prompt['content'],
                        "question_id": question['qid'], "question_language": question['language'],
                        "messages": messages, "response": "", "status": "error", "error": error_msg
                    }
                    print(f"Error: {error_msg}")
                    success = True
            
        all_results.append(result)
        
        if len(all_results) % 5 == 0:
            save_checkpoint(question_idx, prompt_idx + 1, all_results)
        
        time.sleep(1)

# Finalization
save_final_results(all_results, system_prompts_config, questions_config)
print(f"\n--- Test for {TARGET_MODEL} completed! ---")
print(f"Results saved to {RESULTS_FILE}")
successful_tests = sum(1 for result in all_results if result['status'] == 'success')
failed_tests = len(all_results) - successful_tests
print(f"Successful: {successful_tests}, Failed: {failed_tests}")