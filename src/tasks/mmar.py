import os
import numpy as np
from torch.utils.data import Dataset
import datasets
from huggingface_hub import hf_hub_download
import json
import librosa
from tqdm import tqdm
from scipy.io import wavfile
import tarfile
from src import Define
from .utils import extract_mcqa_answer, llm_as_judge, LLMJudgeWrapper

class MMAR(object):

    def __init__(self):
        self.cache_dir = f"{Define.CACHE_DIR}/MMAR"
        if not os.path.exists(self.cache_dir):
            self.parse()
        with open(f"{self.cache_dir}/data_info.json", "r", encoding="utf-8") as f:
            self.info = json.load(f)

    def parse(self):
        hf_cache_dir = f"{Define.CACHE_DIR}/huggingface/mmar"
        if not os.path.exists(hf_cache_dir):
            downloaded_path = hf_hub_download(
                repo_id="BoJack/MMAR",
                filename="mmar-audio.tar.gz",
                repo_type="dataset",
                use_auth_token=True,
                local_dir=hf_cache_dir,
            )
            print(f"✅ Saved to: {downloaded_path}")
            
            # Extract in place
            extract_dir = os.path.dirname(downloaded_path)
            with tarfile.open(downloaded_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)

            print(f"🎉 Extracted in: {extract_dir}")

        src_dataset = datasets.load_dataset("BoJack/MMAR", split="test")
        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        res = []
        for idx, instance in tqdm(enumerate(src_dataset)):
            basename = instance["audio_path"].split('/')[-1][:-4]
            if instance['id'] != basename:
                print(f"Fix id: {instance['id']} -> {basename}.")
                instance['id'] = basename
            wav, _ = librosa.load(f"{hf_cache_dir}/audio/{instance['id']}.wav", sr=16000)
            wavfile.write(f"{self.cache_dir}/wav/{instance['id']}.wav", 16000, (wav * 32767).astype(np.int16))
            res.append(instance)
        with open(f"{self.cache_dir}/data_info.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4)

    def __len__(self):
        return len(self.info)
    
    def get(self, idx) -> dict:
        instance = self.info[idx]
        audio_input, _ = librosa.load(f"{self.cache_dir}/wav/{instance['id']}.wav", sr=16000)

        return {
            **instance,
            "audio_input": audio_input,
        }


class MMARMCQASequence(Dataset):
    def __init__(self, reasoning: bool=False, llm_judge: bool=False, judge_mode: str="", prompt_mode: str="") -> None:
        self.reasoning = reasoning
        self.llm_judge = llm_judge
        self.corpus = MMAR()
        self.idx_seq = list(range(len(self.corpus)))
        if self.llm_judge:
            if judge_mode == 'api':
                judge_model_name = 'gpt-4o-2024-11-20'
            elif judge_mode == 'local':
                judge_model_name = "microsoft/Phi-3.5-mini-instruct"
            # Initialize LLM here
            self.llm = LLMJudgeWrapper(
                mode=judge_mode,
                model_name=judge_model_name,
                api_key=Define.API_KEY if judge_mode == "api" else None
            )
            self.prompt_mode = prompt_mode

    @property
    def task_description(self):
        if self.reasoning:
            if not self.llm_judge:
                prompt = (
                    "Answer the multiple-choice question. After any step-by-step reasoning, "
                    "put the final answer on the last line exactly in the format 'The answer is x.' (x is a single lowercase letter a, b, c, ...). "
                    "Do not include any other text on that final line."
                )
            else:
                if self.prompt_mode == "cot":
                    prompt = (
                        "Please answer the following multiple-choice question. "
                        "You must think step-by-step to analyze the options first, and then provide the final answer. "
                        "Answer with 'x', where x is the right letter in lowercase. "
                    )
                elif self.prompt_mode == "cot-a":
                    prompt = (
                        "You are a helpful sound assistant that can hear the given audio. "
                        "Based on the audio, please answer the following multiple-choice question. "
                        "You must think step-by-step to analyze the options first, and then provide the final answer. "
                        "Answer with 'x', where x is the right letter in lowercase. "
                    )
                elif self.prompt_mode == "regular":
                    prompt = ""
                elif self.prompt_mode == "regular-a":
                    prompt = (
                        "You are a helpful sound assistant that can hear the given audio. "
                        "Based on the audio, please answer the following multiple-choice question. "
                    )
                elif self.prompt_mode == "direct":
                    prompt = (
                        "Answer the multiple-choice question. Answer with 'x', where x is the right letter in lowercase. Only output the letter of your answer."
                    )
                else:
                    print(f"Prompt mode {self.prompt_mode} not supported.")
                    exit(0)
        else:
            prompt = "Answer the multiple-choice question. Answer with 'X', where X is the right letter in uppercase. Only output the letter of your answer."
        return prompt
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        sample = self.corpus.get(self.idx_seq[idx])

        # full_prompt
        full_prompt = sample['question'].strip()
        for idx, choice in enumerate(sample["choices"]):
            full_prompt += f"\n{chr(65 + idx)}. {choice}"
        try:
            output = chr(65 + sample["choices"].index(sample["answer"]))
        except:  # MMAR is not properly formatted
            if '\n' in sample["answer"]:  # 2 samples have multiple answers
                ans = sample["answer"].split('\n')[0]
                output = chr(65 + sample["choices"].index(ans))
            elif sample['id'] == "BV1P4411677K_00-00-00_00-00-20":
                output = 'A'
            else:
                output = None
                for idx, choice in enumerate(sample["choices"]):
                    if sample["answer"].lower() in choice.lower():
                        output = chr(65 + idx)
                        break
                if output is None:
                    raise ValueError(f"Unable to identify the correct option! ({sample['id']})")

        if self.prompt_mode != "regular":
            inst = {
                "id": sample['id'],
                "audio_input": sample['audio_input'],
                "text_input": self.task_description + "\n\n" + full_prompt,
                "output": output.lower(),
                "audio_path": f"{self.corpus.cache_dir}/wav/{sample['id']}.wav"
            }
        else:
            inst = {
                "id": sample['id'],
                "audio_input": sample['audio_input'],
                "text_input": self.task_description + full_prompt,
                "output": output.lower(),
                "audio_path": f"{self.corpus.cache_dir}/wav/{sample['id']}.wav"
            }
        return inst
    
    def extract_answer(self, response: str):
        return extract_mcqa_answer(response=response, key="The answer is" if self.reasoning else None)
    
    def eval(self, pred: str, gt: str, question: str = "") -> float:
        if not self.llm_judge:
            ans = self.extract_answer(pred)
            return float(ans == gt)
        else:
            return llm_as_judge(pred=pred, gt=gt, llm=self.llm, question=question)


class MMARMCQA_RSequence(Dataset):
    def __new__(cls):
        return MMARMCQASequence(reasoning=True)
