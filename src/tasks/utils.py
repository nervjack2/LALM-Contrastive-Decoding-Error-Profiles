import re

import os
import re
import torch
from openai import OpenAI
from transformers import pipeline
from torch.utils.data import Dataset


class LLMJudgeWrapper:
    """
    A wrapper to unify API-based (OpenAI) and Local (HuggingFace) LLM interactions.
    """
    def __init__(self, mode: str = "api", model_name: str = "gpt-4o-2024-11-20", api_key: str = None):
        self.mode = mode
        self.model_name = model_name
        
        if self.mode == "api":
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            self.system_prompt = "You are a good judge. You will be given a question with list of possible options, a ground truth answer and a model generated response. You have to determine whether the model generated answer is correct."
        
        elif self.mode == "local":
            print(f"Loading local model: {model_name}...")
            try:
                self.pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    model_kwargs={"torch_dtype": torch.float16},
                    device_map="auto",
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load local model {model_name}: {e}")

    def generate(self, user_prompt: str) -> str:
        if self.mode == "api":
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.0,
                top_p=1.0
            )
            return response.choices[0].message.content
        
        elif self.mode == "local":
            messages = [
                {"role": "system", "content": "You are a helpful assistant and a strict judge."},
                {"role": "user", "content": user_prompt}
            ]
            outputs = self.pipe(messages, max_new_tokens=512, do_sample=False)
            return outputs[0]["generated_text"][-1]["content"]
        
        return ""

def llm_as_judge(pred: str, gt: str, llm: LLMJudgeWrapper, question: str = "") -> float:
    """
    Evaluates the prediction using an LLM.
    Returns 1.0 if correct, 0.0 otherwise.
    """
    user_prompt = f'''
    You will be given a question with list of possible options, a ground truth answer and a model generated response. Determine whether the model generated response is correct based on the following criteria:
    1. Since there is one and only one corect answer, it should be judged incorrect if the model do not choose any option from the option list or it choose more than one option.
    2. If the model choose one option from the option list, it should be judged correct if the chosen option aligns with the ground truth answer, otherwise it should be judged incorrect.
    3. The judgment should focus on the final choice made by the model. Ignore any expressions of uncertainty, hesitation, or disclaimers (e.g., "I guess," "I think," "It might be," "Based on limited info"). As long as a specific option is selected, evaluate it against the ground truth.
    4. Read the question, options, ground truth answer and model generated response carefully before making a decision.

    Considering the following examples:
    Question: What is the capital of France? (a) Paris (b) London (c) Berlin (d) Madrid
    Ground truth answer: (a) Paris
    If the model generated response is: "The capital of France is Tokyo.", it should be judged incorrect since it does not choose any option from the option list.
    If the model generated response is: "The capital of France is Paris and London.", it should be judged incorrect since it chooses more than one option from the option list.
    If the model generated response is: "The capital of France is Paris.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
    If the model generated response is: "I am not entirely sure, but based on the audio, I'd say the answer is a. It is just a guess.", it should be judged correct. Even though the model expresses uncertainty.
    Another Question: What is the underlying emotion of the speaker? (a) Happy (b) Sad (c) Angry (d) Neutral
    Ground truth answer: (a) Happy
    If the model generated response is: "The speaker is happy.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
    If the model generated response is: "The speaker expresses happiness.", it should be judged correct since "happiness" aligns with the ground truth answer "happy", and they are just different part of speech of the same word.
    
    Now here is the question and the model generated response for you to judge:
    Question: {question}
    Ground truth answer: {gt}
    Model generated response: {pred}

    Carefully make your decision based on the above criteria. Return your judgement with the following format:
    Explanation: <Your explanation on your judgement>
    Judgement: <Your judgement, either "correct" or "incorrect">
    '''
    output = llm.generate(user_prompt)
    print(output)

    pattern = r"Judgement:\s*(correct|incorrect)"
    match = re.search(pattern, output, re.IGNORECASE)
    
    if match:
        result = match.group(1).lower()
        return 1.0 if result == "correct" else 0.0
    else:
        print(f"[Warning] Regex failed on LLM output: {output[:100]}...")
        return 0.0


def extract_mcqa_answer(response: str, key: str | None = None):
    """
    Rule-based extraction for the final answer from a model response.
    
    If `key` is provided (e.g., 'The answer is', 'Final choice', etc.),
    searches for that key followed by a single lowercase letter (a, b, c, ...),
    possibly with punctuation such as '.', ':', or ')'.

    If no key is provided, assumes the response itself starts with the answer.
    """
    text = response.strip().lower()

    if key is not None:
        key_pattern = re.escape(key.lower())

        pattern = rf"{key_pattern}[:\s\(\[]*([a-z])[\.\)\]]?\b"

        match = re.search(pattern, text)
        if match:
            return match.group(1)

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in reversed(lines):
            if re.fullmatch(r"[a-z]\.?", line):
                return line[0]
        return ""
    return response[0].lower()


def extract_yes_no_answer(response: str, key: str | None = None):
    """
    Rule-based extraction for a Yes/No answer from model response text.
    
    If `key` is provided (e.g. 'The answer is'), searches for that key followed
    by 'yes' or 'no', possibly with punctuation like '.' or ':'.
    
    If no key is provided, looks for a standalone 'yes' or 'no' at the end
    or start of the response.
    """
    text = response.strip().lower()

    if key is not None:
        key_pattern = re.escape(key.lower())

        pattern = rf"{key_pattern}[:\s]+(yes|no)[\.\!]?(\b|$)"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        matches = re.findall(r"\b(yes|no)\b", text)
        if matches:
            return matches[-1]
        return ""
    else:
        if text.startswith("yes"):
            return "yes"
        if text.startswith("no"):
            return "no"
        if text.endswith("yes"):
            return "yes"
        if text.endswith("no"):
            return "no"
        return ""
