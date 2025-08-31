import json
from together import Together
from datetime import datetime
import time
import os
import signal
import sys
import argparse

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

def signal_handler(signum, frame):
    global shutdown_requested
    print("\n\nShutdown requested. Saving progress...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run bulk testing with optional model skipping and temperature control')
parser.add_argument('-skipm', '--skip-models', type=int, default=0, 
                    help='Number of models to skip from the beginning (default: 0)')
parser.add_argument('-t', '--temperature', type=float, default=1,
                    help='Set the temperature for the LLM response (default: 1)')
args = parser.parse_args()

def save_checkpoint(current_model_idx, current_question_idx, current_prompt_idx, all_results):
    checkpoint_data = {
        'current_model_idx': current_model_idx,
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

client = Together()

# Load checkpoint or start fresh
checkpoint = load_checkpoint()
if checkpoint:
    print(f"Resuming from checkpoint saved at {checkpoint['timestamp']}")
    all_results = checkpoint['completed_results']
    if args.skip_models > 0:
        start_model_idx, start_question_idx, start_prompt_idx = args.skip_models, 0, 0
        print(f"Command line override: skipping to model {args.skip_models + 1}")
    else:
        start_model_idx = checkpoint['current_model_idx']
        start_question_idx = checkpoint['current_question_idx']
        start_prompt_idx = checkpoint['current_prompt_idx']
else:
    print("Starting fresh bulk testing...")
    all_results = []
    start_model_idx, start_question_idx, start_prompt_idx = args.skip_models, 0, 0

if args.skip_models > 0:
    if args.skip_models >= len(model_config['models']):
        print(f"Error: Cannot skip {args.skip_models} models. Only {len(model_config['models'])} models available.")
        sys.exit(1)
    print(f"Skipping first {args.skip_models} model(s). Starting from model {args.skip_models + 1}")

if start_model_idx >= len(model_config['models']):
    print("All models completed!")
    sys.exit(0)

print(f"Starting bulk testing with {len(model_config['models'])} models, {len(system_prompts_config['system_prompts'])} system prompts, and {len(questions_config['questions'])} questions...")
print(f"Using temperature: {args.temperature}") # Inform the user of the temperature setting

# Main testing loop
for model_idx, model in enumerate(model_config['models']):
    if model_idx < start_model_idx:
        continue

    rate_limit = MODEL_RATE_LIMITS.get(model, None)
    requests_made, window_start = 0, time.time()
        
    for question_idx, question in enumerate(questions_config['questions']):
        if model_idx == start_model_idx and question_idx < start_question_idx:
            continue
            
        matching_prompts = [sp for sp in system_prompts_config['system_prompts'] if sp['language'] == question['language']]
        
        if not matching_prompts:
            print(f"Warning: No matching system prompts for question {question['qid']} (lang: {question['language']})")
            continue
            
        for prompt_idx, system_prompt in enumerate(matching_prompts):
            if model_idx == start_model_idx and question_idx == start_question_idx and prompt_idx < start_prompt_idx:
                continue
                
            if shutdown_requested:
                save_checkpoint(model_idx, question_idx, prompt_idx, all_results)
                print(f"Checkpoint saved. Progress: Model {model_idx+1}/{len(model_config['models'])}, Question {question_idx+1}/{len(questions_config['questions'])}, Prompt {prompt_idx+1}/{len(matching_prompts)}")
                sys.exit(0)
            
            print(f"Testing model: {model} | System prompt: {system_prompt['id']} | Question: {question['qid']} ({question['language']})")
            
            messages = [{"role": "system", "content": system_prompt['content']}, {"role": "user", "content": question['content']}]
            
            success = False
            while not success:
                try:
                    # --- MODIFICATION: The temperature parameter is added here ---
                    stream = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True,
                        temperature=args.temperature, # Use the value from the command line
                    )
                    full_response = ""
                    for chunk in stream:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                                content = chunk.choices[0].delta.content or ""
                                full_response += content
                    result = {
                        "timestamp": datetime.now().isoformat(), "model": model,
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
                                print(f"Rate limit reached for {model}. Waiting {int(wait_time)+1} seconds...")
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
                        print(f"Rate limit error for {model}. Waiting {int(wait_seconds)} seconds and retrying...")
                        time.sleep(wait_seconds)
                        if rate_limit: window_start, requests_made = time.time(), 0
                    else:
                        result = {
                            "timestamp": datetime.now().isoformat(), "model": model,
                            "system_prompt_id": system_prompt['id'], "system_prompt_content": system_prompt['content'],
                            "question_id": question['qid'], "question_language": question['language'],
                            "messages": messages, "response": "", "status": "error", "error": error_msg
                        }
                        print(f"Error: {error_msg}")
                        success = True
                
            all_results.append(result)
            
            if len(all_results) % 5 == 0:
                save_checkpoint(model_idx, question_idx, prompt_idx + 1, all_results)
            
            time.sleep(1)

# Finalization
save_final_results(all_results, model_config, system_prompts_config, questions_config)
print(f"\nBulk testing completed! Results saved to {RESULTS_FILE}")
successful_tests = sum(1 for result in all_results if result['status'] == 'success')
failed_tests = len(all_results) - successful_tests
print(f"Successful: {successful_tests}, Failed: {failed_tests}")