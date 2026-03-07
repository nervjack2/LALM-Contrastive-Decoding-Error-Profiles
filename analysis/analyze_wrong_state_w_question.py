import os
import json
import sys 
import re
from typing import Dict
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.tasks.load import get_test_task
from src import Define
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    pass

# --- Configuration ---
# Define states
STATES_WRONG = [
    "W_Hallucinated_No_Audio",
    "W_Reasoning_but_Wrong_Answer", 
    "W_Direct_Assertive_Wrong_Answer", 
    "W_Guessing_Wrong_or_Refusal"
]

# 簡化 Correct 狀態，不再細分
STATES_CORRECT = [
    "Correct_Response" 
]

SINGLE_STATE_SYSTEM_PROMPT = """
You are an expert analyst evaluating the response state of AI models in audio tasks.
You will be provided with ONE text output. The status of this output is confirmed to be WRONG.

Your task is to classify the "State" of the text based on the definitions below. 
You MUST follow the STRICT PRIORITY ORDER (1 -> 2 -> 3 -> 4). 
Assign the FIRST state that fits the description and ignore subsequent categories.

### Priority Evaluation Process (Check in this order):

1. **[Priority 1] W_Hallucinated_No_Audio**
   - Check this FIRST.
   - Definition: The model falsely claims there is no audio provided or asks the user to provide the audio.
   - Includes: 
     - "no audio provided", "I can't hear", "play the audio for me", "text-only input".
     - "I can't tell from just this text".
   - Exclusion: If the model says "I can't tell **from this audio**" (acknowledging presence) or "The audio is unclear", DO NOT select P1.
   - Note: If the model hears the audio but calls it "unclear/vague", DO NOT select this (go to P4).
   - If YES -> Stop and assign this state.

2. **[Priority 2] W_Reasoning_but_Wrong_Answer**
   - Check this SECOND.
   - Definition: The model provides **specific acoustic or semantic evidence** to support a **wrong definitive answer**.
   - **CRITICAL EXCLUSIONS (Do NOT select P2 for these, skip to P3 or P4):**
     1. **Circular/Restatement:** "It is male, so the answer is male." (No new evidence provided -> Go to P3).
     2. **Simple Sensory:** "It sounds like male." (No specific justification -> Go to P3).
     3. **Reasoning towards Refusal:** "The audio is too short to tell, so I can't answer." (Conclusion is refusal -> Go to P4).
   - If YES (has specific evidence + definitive wrong conclusion) -> Stop and assign this state.

3. **[Priority 3] W_Direct_Assertive_Wrong_Answer**
   - Check this THIRD.
   - Definition: The model asserts a wrong answer confidently, neutrally, or based on simple sensory intuition WITHOUT specific evidence.
   - Includes:
     - Short assertions: "Male.", "Option B."
     - Restatements: "The speaker is male, so select Male."
     - Sensory intuition: "It sounds like a male voice." (without explaining *why*).
   - If YES -> Stop and assign this state.

4. **[Priority 4] W_Guessing_Wrong_or_Refusal**
   - Check this LAST.
   - Definition: The model explicitly states it is "not sure", "guessing", or refuses to answer.
   - Note: Includes cases where the model reasons about the input but concludes it has insufficient information.
   - If YES -> Assign this state.

### Output Format (JSON)
{
  "state": "CATEGORY_NAME_FROM_LIST",
  "reason": "Brief explanation of why you assigned this state."
}
"""

class LLMJudgeWrapper:
    def __init__(self, mode: str = "api"):
        self.mode = mode
        
        if self.mode == "api":
            self.model_name = "gpt-4o-2024-11-20"
            if OpenAI is None:
                raise ImportError("Please install openai package: pip install openai")
            self.client = OpenAI(api_key=Define.API_KEY)
        elif self.mode == "local":
            self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            print(f"--- Loading Local Model: {self.model_name} on A6000 (CUDA) ---")
            # 檢查 GPU
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Please check your GPU settings.")
            
            # Load Tokenizer & Model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            
            # Setup Pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            print("--- Model Loaded Successfully ---")

    def analyze_single(self, text: str, score: float, question: str) -> dict:
        if score == 1.0:
            return {
                "state": "Correct_Response", 
                "reason": "Score is 1.0. Correct responses are not subdivided."
            }

        status = "WRONG"

        user_prompt = f"""
        Please analyze this response:
        
        Status: {status}
        Question: {question}
        Response: {text}
        
        Assign the response state strictly based on the Definitions provided.
        """

        raw_response = ""
        if self.mode == "api":
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SINGLE_STATE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model_name,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                raw_response = response.choices[0].message.content
            except Exception as e:
                return {"state": "Error", "reason": str(e)}
            
        elif self.mode == "local":
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Generate
                outputs = self.pipe(
                    prompt,
                    max_new_tokens=512,
                    do_sample=False,   # Greedy decoding for consistency
                    temperature=0.0,
                    return_full_text=False
                )
                raw_response = outputs[0]["generated_text"]
            except Exception as e:
                return {"state": "Error", "reason": f"Local Generation Error: {str(e)}"}
        
        try:
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"state": "Error", "reason": "No JSON found"}
        except json.JSONDecodeError:
            return {"state": "Error", "reason": "JSON Decode Error"}

def parse_prediction_file(filepath: str) -> Dict[str, dict]:
    """Parses score|id|prediction|text format."""
    data = {}
    current_id = None
    header_pattern = re.compile(r'^(\d+\.\d+)\|([^|]+)\|([^|]+)\|(.*)')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = header_pattern.match(line)
            if match:
                score_str, uid, prediction, text_start = match.groups()
                try:
                    score = float(score_str)
                    data[uid] = {'score': score, 'prediction': prediction, 'text': text_start.strip()}
                    current_id = uid
                except ValueError:
                    continue
            else:
                if current_id:
                    stripped = line.strip()
                    if stripped: data[current_id]['text'] += '\n' + stripped
                    else: data[current_id]['text'] += '\n'
    return data

def main_analyze_file(input_path, output_jsonl, task_name, mode):
    print(f"1. Loading File: {input_path}")
    data = parse_prediction_file(input_path)
    print(f"   Found {len(data)} samples.")

    question_dict = {}
    ds = get_test_task(task_name)
    for sample in ds:
        question, _id = sample["text_input"], sample["id"]
        question_dict[_id] = question

    llm = LLMJudgeWrapper(mode=mode)
    
    processed_ids = set()
    
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                processed_ids.add(entry['id'])
    
    print(f"   Already processed {len(processed_ids)} samples. Resuming..." if processed_ids else "   Starting fresh...")

    with open(output_jsonl, 'a', encoding='utf-8') as f_out:
        for i, (file_id, entry) in enumerate(data.items()):
            if file_id in processed_ids:
                continue

            question = question_dict[file_id]
            if i % 10 == 0:
                print(f"   Processing {i+1}/{len(data)}...")
            
            analysis = llm.analyze_single(entry['text'], entry['score'], question)
     
            record = {
                "id": file_id,
                "score": entry['score'],
                "text": entry['text'],
                "state": analysis.get("state", "Error"),
                "reason": analysis.get("reason", "")
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush() 

    print(f"Analysis Complete. Results saved to {output_jsonl}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_state.py <input_result.txt> <output_states.jsonl> <task_name> <mode>")
        sys.exit(1)

    INPUT_FILE = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]
    TASK_NAME = sys.argv[3]
    MODE = sys.argv[4]
    
    if os.path.exists(INPUT_FILE):
        main_analyze_file(INPUT_FILE, OUTPUT_FILE, TASK_NAME, MODE)
    else:
        print(f"File not found: {INPUT_FILE}")