import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from src.systems.generation.logits_process import VCDLogitsProcessor
from .desta2_5 import Desta2_5System


class ACDSystem(Desta2_5System):
    def __init__(self, config):
        super().__init__(config)
        # Load hyperparameters from config
        vcd_config = self.model_config["vcd"]
        self.alpha = vcd_config["alpha"]
        self.beta = vcd_config["beta"]
        # Noise level gamma. Paper uses diffusion steps, we approximate with direct noise injection.
        self.noise_gamma = vcd_config["gamma"]
        self.noise_steps = vcd_config["steps"]

    def apply_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian noise to the audio input via a multi-step diffusion process.
        """
        # Ensure audio is float for noise addition
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32) / 32768.0 # Normalize if PCM16

        # Initialize v_0
        v_0 = audio.copy()

        # Apply the closed-form formula
        alpha = 1 - self.noise_gamma
        alpha_bar = alpha ** self.noise_steps
        epsilon = np.random.normal(0, 1, v_0.shape)
        v_t = np.sqrt(alpha_bar) * v_0 + np.sqrt(1 - alpha_bar) * epsilon 
        
        return v_t
    
    def prepare_logits_processor(
        self,
        audios: list[np.ndarray],
        transcriptions: list[str],
        texts: list[str],
    ) -> VCDLogitsProcessor:
        # create negative context
        audio_neg = [self.apply_noise(a) for a in audios]
        transcription_neg = self.model.prepare_transcriptions(audio_neg, [None])
        prompts_neg = [
            self.format_prompt(audio, transcription, text)
        for (audio, transcription, text) in zip(audio_neg, transcription_neg, texts)]
        
        inputs_distorted = self.processor(
            audio=audio_neg,
            transcription=transcription_neg, 
            text=prompts_neg,
            add_special_tokens=False,
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        logits_processor = VCDLogitsProcessor(
            model=self.model,
            inputs_distorted=inputs_distorted,
            alpha=self.alpha,
            beta=self.beta
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
        acd_logits_processor = self.prepare_logits_processor(audios, transcriptions, texts)
        
        # 2. Prepare Main Inputs
        inputs = self.processor(
            audio=audios,
            transcription=transcriptions, 
            text=prompts,
            add_special_tokens=False,
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # 3. Generate with ACD
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=[acd_logits_processor]
        )
        
        # Decode prediction
        prediction = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Cleanup to prevent VRAM leaks
        del acd_logits_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction,
        }
