import json
from together import Together
from datetime import datetime
import time

# Load model configuration
with open('models.json', 'r') as f:
    model_config = json.load(f)

# Load questions
with open('questions.json', 'r') as f:
    questions_config = json.load(f)

# Load system prompts
with open('system_prompts.json', 'r') as f:
    system_prompts_config = json.load(f)

client = Together()

# Store all results
all_results = []

print(f"Starting bulk testing with {len(model_config['models'])} models, {len(system_prompts_config['system_prompts'])} system prompts, and {len(questions_config['questions'])} questions...")

# Test each combination: model x filtered_system_prompts x question
for model in model_config['models']:
    for question in questions_config['questions']:
        # Filter system prompts that match the question's prompt_type and language
        matching_prompts = [
            sp for sp in system_prompts_config['system_prompts'] 
            if sp['prompt_type'] == question['prompt_type'] and sp['language'] == question['language']
        ]
        
        if not matching_prompts:
            print(f"Warning: No matching system prompts found for question {question['qid']} (type: {question['prompt_type']}, lang: {question['language']})")
            continue
            
        for system_prompt in matching_prompts:
            print(f"Testing model: {model} | System prompt: {system_prompt['id']} | Question: {question['qid']} ({question['language']}, {question['prompt_type']})")
            
            # Collect the full response
            full_response = ""
            
            # Prepare messages with system prompt
            messages = [{"role": "system", "content": system_prompt['content']}] + question['messages']
            
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

                # Store the result
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "system_prompt_id": system_prompt['id'],
                    "system_prompt_content": system_prompt['content'],
                    "question_id": question['qid'],
                    "question_topic": question['topic'],
                    "question_prompt_type": question['prompt_type'],
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
                    "question_topic": question['topic'],
                    "question_prompt_type": question['prompt_type'],
                    "question_language": question['language'],
                    "messages": messages,
                    "response": "",
                    "status": "error",
                    "error": str(e)
                }
                print(f"Error: {str(e)}")
                
            all_results.append(result)
            
            # Small delay between requests
            time.sleep(1)

# Save all results to JSON 
output_data = {
    "test_summary": {
        "total_tests": len(all_results),
        "models_tested": model_config['models'],
        "system_prompts_available": [sp['id'] for sp in system_prompts_config['system_prompts']],
        "questions_tested": [q['qid'] for q in questions_config['questions']],
        "prompt_types": list(set(q['prompt_type'] for q in questions_config['questions'])),
        "languages": list(set(q['language'] for q in questions_config['questions'])),
        "timestamp": datetime.now().isoformat()
    },
    "results": all_results
}

with open('bulk_test_results.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nBulk testing completed! Results saved to bulk_test_results.json")
print(f"Total tests run: {len(all_results)}")

# Print summary 
successful_tests = sum(1 for result in all_results if result['status'] == 'success')
failed_tests = len(all_results) - successful_tests
print(f"Successful: {successful_tests}, Failed: {failed_tests}")
