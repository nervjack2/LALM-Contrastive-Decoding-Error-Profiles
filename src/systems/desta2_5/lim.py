import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from src.systems.generation.logits_process import LIMLogitsProcessor
from .desta2_5 import Desta2_5System


class LIMSystem(Desta2_5System):
    def __init__(self, config):
        super().__init__(config)
        # Load hyperparameters from config
        lim_config = self.model_config["lim"]
        self.omega = lim_config["omega"]
        self.tau = lim_config["tau"]
        self.negative_prompt = lim_config["negative_prompt"]
        self.negative_prompt_tokens = self.processor(
            text=[self.negative_prompt],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(self.device)["input_ids"]
    
    def prepare_logits_processor(
        self,
        audios: list[np.ndarray],
        transcriptions: list[str],
        texts: list[str],
    ) -> LIMLogitsProcessor:
        # LIM shares the same context by design
        prompts = [
            self.format_prompt(audio, transcription, text)
        for (audio, transcription, text) in zip(audios, transcriptions, texts)]

        inputs = self.processor(
            audio=audios,
            transcription=transcriptions, 
            text=prompts,
            add_special_tokens=False,
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        logits_processor = LIMLogitsProcessor(
            model=self.model,
            inputs=inputs,
            negative_prompt_tokens=self.negative_prompt_tokens,
            omega=self.omega,
            tau=self.tau
        )
        
        return logits_processor

    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> dict:
        assert len(texts) == 1, "Currently no batch inference"

        # prepare inputs
        transcriptions = self.model.prepare_transcriptions(audios, [None])
        # print(transcriptions)

        prompts = [
            self.format_prompt(audio, transcription, text)
        for (audio, transcription, text) in zip(audios, transcriptions, texts)]

        # 1. Prepare Logits Processor
        lim_logits_processor = self.prepare_logits_processor(audios, transcriptions, texts)
        
        # 2. Prepare Main Inputs
        inputs = self.processor(
            audio=audios,
            transcription=transcriptions, 
            text=prompts,
            add_special_tokens=False,
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # 3. Generate with LIM
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=[lim_logits_processor]
        )
        
        # Decode prediction
        prediction = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Cleanup to prevent VRAM leaks
        del lim_logits_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction,
        }
