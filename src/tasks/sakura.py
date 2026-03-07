import os
import numpy as np
from torch.utils.data import Dataset
import datasets
import json
import librosa
from tqdm import tqdm
from scipy.io import wavfile
import random

from src import Define
from .utils import llm_as_judge, LLMJudgeWrapper

class SAKURA(object):

    HF_MAPPING = {
        "animal": "SLLM-multi-hop/AnimalQA",
        "emotion": "SLLM-multi-hop/EmotionQA",
        "gender": "SLLM-multi-hop/GenderQA",
        "language": "SLLM-multi-hop/LanguageQA",
    }

    def __init__(self, subject: str, multi: bool=False):
        self.cache_dir = f"{Define.CACHE_DIR}/SAKURA/{subject}"
        self.subject = subject
        self.multi = multi
        self.data_info_path = f"{self.cache_dir}/data_info.json" if not multi else f"{self.cache_dir}/data_info-multi.json"
        if not os.path.isfile(self.data_info_path):
            self.parse()
        with open(self.data_info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)
        
        self.audio_input_paths = []
        self.text_inputs = []
        self.outputs = []
        for item in self.info:
            self.audio_input_paths.append(f"{self.cache_dir}/wav/{item['audio_input_path']}")
            self.text_inputs.append(item["text_input"])
            self.outputs.append(item["output"])
    
    def parse(self):
        src_dataset = datasets.load_dataset(SAKURA.HF_MAPPING[self.subject], split="test")

        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        res = []
        for idx, instance in tqdm(enumerate(src_dataset)):
            if not os.path.isfile(f"{self.cache_dir}/wav/{instance['file']}"):
                wav = librosa.resample(
                    instance["audio"]["array"],
                    orig_sr=instance["audio"]["sampling_rate"],
                    target_sr=16000
                )
                wavfile.write(f"{self.cache_dir}/wav/{instance['file']}", 16000, (wav * 32767).astype(np.int16))
            if self.multi:
                res.append({
                    "audio_input_path": instance['file'],
                    "text_input": instance["multi_instruction"],
                    "output": instance["multi_answer"].strip()[1],
                    "attribute_label": instance["multi_answer"]
                })
            else:
                res.append({
                    "audio_input_path": instance['file'],
                    "text_input": instance["single_instruction"],
                    "output": instance["single_answer"].strip()[1],
                    "attribute_label": instance["attribute_label"]
                })
        with open(self.data_info_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4)

    def __len__(self):
        return len(self.audio_input_paths)
    
    def get(self, idx) -> dict:
        audio_input, _ = librosa.load(self.audio_input_paths[idx], sr=16000)

        return {
            "id": self.audio_input_paths[idx],
            "audio_input": audio_input,
            "text_input": self.text_inputs[idx],
            "output": self.outputs[idx],
            "audio_path": self.audio_input_paths[idx]
        }


class SAKURASequence(Dataset):
    def __init__(self, subject: str, multi: bool=False, judge_mode: str=""):
        self.corpus = SAKURA(subject=subject, multi=multi)
        self.idx_seq = list(range(len(self.corpus)))
        random.shuffle(self.idx_seq)
        # print(self.idx_seq[:5])
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

    @property
    def task_description(self):
        return ""
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        inst = self.corpus.get(self.idx_seq[idx])
        full_prompt = inst["text_input"]
        inst["text_input"] = self.task_description + full_prompt
        return inst
    
    def eval(self, pred: str, gt: str, question: str = "") -> float:
        """
        Modified eval to accept 'question' which is critical for the LLM Judge prompt.
        """
        return llm_as_judge(pred=pred, gt=gt, llm=self.llm, question=question)