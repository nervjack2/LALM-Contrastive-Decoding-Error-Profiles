import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from src.systems.generation.logits_process import AADLogitsProcessor
from .desta2_5 import Desta2_5System


class AADSystem(Desta2_5System):
    def __init__(self, config):
        super().__init__(config)
        aad_config = self.model_config.get("aad", {})
        self.alpha = aad_config.get("alpha", 0.5)
        self.threshold = aad_config.get("threshold", -1) 

    def prepare_logits_processor(
        self,
        audios: list[np.ndarray],
        transcriptions: list[str],
        texts: list[str],
        negative_prompt: str="none"
    ) -> AADLogitsProcessor:
        # create negative context
        if negative_prompt == "none":
            audio_neg, transcription_neg = None, None
            prompts_neg = [
                self.format_prompt(None, None, text)
            for text in texts]
        elif negative_prompt == "silence":
            audio_neg = [np.zeros_like(x) for x in audios]
            transcription_neg = [" "] * len(audio_neg)
            prompts_neg = [
                self.format_prompt(audio, transcription, text)
            for (audio, transcription, text) in zip(audio_neg, transcription_neg, texts)]
        else:
            raise NotImplementedError
        
        inputs_without_audio = self.processor(
            audio=audio_neg,
            transcription=transcription_neg, 
            text=prompts_neg,
            add_special_tokens=False,
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        logits_processor = AADLogitsProcessor(
            model=self.model,
            inputs_without_audio=inputs_without_audio,
            alpha=self.alpha,
            threshold=self.threshold,
        )
        
        return logits_processor

    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> dict:
        assert len(texts) == 1, "Currently no batch inference"
        
        transcriptions = self.model.prepare_transcriptions(audios, [None])

        prompts = [
            self.format_prompt(audio, transcription, text)
        for (audio, transcription, text) in zip(audios, transcriptions, texts)]

        # 1. Prepare Logits Processor (The "Without Audio" Path)
        aad_logits_processor = self.prepare_logits_processor(audios, transcriptions, texts)
        
        # 2. Prepare Main Inputs (The "With Audio" Path)
        inputs = self.processor(
            audio=audios,
            transcription=transcriptions, 
            text=prompts,
            add_special_tokens=False,
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # 3. Generate with AAD
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=[aad_logits_processor]
        )
        
        prediction = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        step_all, step_applied = aad_logits_processor.call_cnt, aad_logits_processor.apply_cnt
        
        del aad_logits_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction,
            "steps": (step_all, step_applied)
        }
