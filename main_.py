import json
from together import Together
from datetime import datetime
import time
import os
import signal
import sys

# Checkpoint file to save progress
CHECKPOINT_FILE = 'bulk_test_checkpoint.json'
RESULTS_FILE = 'test_results.json'

# Global variable to handle graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    global shutdown_requested
    print("\n\nShutdown requested. Saving progress...")
    shutdown_requested = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def save_checkpoint(current_model_idx, current_question_idx, current_prompt_idx, all_results):
    """Save current progress to checkpoint file"""
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
    """Load checkpoint if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except:
            print("Warning: Could not load checkpoint file. Starting from beginning.")
            return None
    return None

def save_final_results(all_results, model_config, system_prompts_config, questions_config):
    """Save final results and clean up checkpoint"""
    output_data = {
        "test_summary": {
            "total_tests": len(all_results),
            "models_tested": model_config['models'],
            "system_prompts_available": [sp['id'] for sp in system_prompts_config['system_prompts']],
            "questions_tested": [q['qid'] for q in questions_config['questions']],
            # The 'prompt_types' field has been removed as it is no longer in the questions file.
            "languages": list(set(q['language'] for q in questions_config['questions'])),
            "timestamp": datetime.now().isoformat()
        },
        "results": all_results
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Remove checkpoint file since we're done
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

# Load model configuration
with open('models.json', 'r', encoding='utf-8') as f:
    model_config = json.load(f)

# Load questions
with open('questions.json', 'r', encoding='utf-8') as f:
    questions_config = json.load(f)

# Load system prompts
with open('system_prompts.json', 'r', encoding='utf-8') as f:
    system_prompts_config = json.load(f)

client = Together()

# Load checkpoint if it exists
checkpoint = load_checkpoint()
if checkpoint:
    print(f"Resuming from checkpoint saved at {checkpoint['timestamp']}")
    print(f"Already completed {len(checkpoint['completed_results'])} tests")
    all_results = checkpoint['completed_results']
    start_model_idx = checkpoint['current_model_idx']
    start_question_idx = checkpoint['current_question_idx']
    start_prompt_idx = checkpoint['current_prompt_idx']
else:
    print("Starting fresh bulk testing...")
    all_results = []
    start_model_idx = 0
    start_question_idx = 0
    start_prompt_idx = 0

print(f"Starting bulk testing with {len(model_config['models'])} models, {len(system_prompts_config['system_prompts'])} system prompts, and {len(questions_config['questions'])} questions...")

# Test each combination: model x filtered_system_prompts x question
for model_idx, model in enumerate(model_config['models']):
    # Skip models we've already processed
    if model_idx < start_model_idx:
        continue
        
    for question_idx, question in enumerate(questions_config['questions']):
        # Skip questions we've already processed for this model
        if model_idx == start_model_idx and question_idx < start_question_idx:
            continue
            
        # Filter system prompts that match the question's language (prompt_type is removed)
        matching_prompts = [
            sp for sp in system_prompts_config['system_prompts'] 
            if sp['language'] == question['language']
        ]
        
        if not matching_prompts:
            print(f"Warning: No matching system prompts found for question {question['qid']} (lang: {question['language']})")
            continue
            
        for prompt_idx, system_prompt in enumerate(matching_prompts):
            # Skip prompts we've already processed for this model/question combination
            if model_idx == start_model_idx and question_idx == start_question_idx and prompt_idx < start_prompt_idx:
                continue
                
            # Check for shutdown request
            if shutdown_requested:
                save_checkpoint(model_idx, question_idx, prompt_idx, all_results)
                print(f"Checkpoint saved. Progress: Model {model_idx+1}/{len(model_config['models'])}, Question {question_idx+1}/{len(questions_config['questions'])}, Prompt {prompt_idx+1}/{len(matching_prompts)}")
                print("You can resume by running the script again.")
                sys.exit(0)
            
            print(f"Testing model: {model} | System prompt: {system_prompt['id']} | Question: {question['qid']} ({question['language']})")
            
            # Collect the full response
            full_response = ""
            
            # Prepare messages with system prompt and the new question format
            # The user role is now hardcoded as it is consistent
            messages = [
                {"role": "system", "content": system_prompt['content']},
                {"role": "user", "content": question['content']}
            ]
            
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )

                for chunk in stream:
                    # Check if choices array exists and has content
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content or ""
                            full_response += content

                # Store the result, removing fields that are no longer available
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "system_prompt_id": system_prompt['id'],
                    "system_prompt_content": system_prompt['content'],
                    "question_id": question['qid'],
                    "question_language": question['language'],
                    "messages": messages,
                    "response": full_response.strip(),
                    "status": "success"
                }
                
            except Exception as e:
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "system_prompt_id": system_prompt['id'],
                    "system_prompt_content": system_prompt['content'],
                    "question_id": question['qid'],
                    "question_language": question['language'],
                    "messages": messages,
                    "response": "",
                    "status": "error",
                    "error": str(e)
                }
                print(f"Error: {str(e)}")
                
            all_results.append(result)
            
            # Save checkpoint every 5 tests to prevent data loss
            if len(all_results) % 5 == 0:
                save_checkpoint(model_idx, question_idx, prompt_idx + 1, all_results)
            
            # Small delay between requests
            time.sleep(1)

# Save all results to JSON 
save_final_results(all_results, model_config, system_prompts_config, questions_config)

print(f"\nBulk testing completed! Results saved to {RESULTS_FILE}")
print(f"Total tests run: {len(all_results)}")

# Print summary 
successful_tests = sum(1 for result in all_results if result['status'] == 'success')
failed_tests = len(all_results) - successful_tests
print(f"Successful: {successful_tests}, Failed: {failed_tests}")