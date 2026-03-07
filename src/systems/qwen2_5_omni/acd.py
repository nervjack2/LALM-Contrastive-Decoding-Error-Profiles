import numpy as np
import torch
import torch.nn.functional as F
from src.systems.generation.logits_process import VCDLogitsProcessor
from copy import deepcopy
from .qwen2_5_omni import Qwen2_5OmniSystem

class ACDSystem(Qwen2_5OmniSystem):
    """
    Implementation of "Mitigating Object Hallucinations in Large Vision-Language Models through 
    Visual Contrastive Decoding" (arXiv:2311.16922v1), adapted for Qwen-Omni.
    """
    def __init__(self, config):
        super().__init__(config)
        # Load hyperparameters from config or use paper defaults 
        self.alpha = self.model_config.get("vcd").get("alpha", 1.0)
        self.beta = self.model_config.get("vcd").get("beta", 0.1)
        # Noise level gamma. Paper uses diffusion steps, we approximate with direct noise injection.
        self.noise_gamma = self.model_config.get("vcd").get("gamma", 0.1) 
        self.noise_steps = self.model_config.get("vcd").get("steps", 100)

    def apply_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian noise to the audio input via a multi-step diffusion process.
        """
        # Ensure audio is float for noise addition
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32) / 32768.0 # Normalize if PCM16
        
        v_0 = audio.copy()

        # Apply the closed-form formula
        alpha = 1 - self.noise_gamma
        alpha_bar = alpha ** self.noise_steps
        epsilon = np.random.normal(0, 1, v_0.shape)
        v_t = np.sqrt(alpha_bar) * v_0 + np.sqrt(1 - alpha_bar) * epsilon 

        return v_t

    def prepare_logits_processor(self, prompts, audios_distorted) -> VCDLogitsProcessor:
        """
        Prepares the secondary inputs (distorted audio) and initializes the processor.
        """
        # Encode the distorted input (v', x)
        # We use the same prompt text, but with noisy audio
        inputs_distorted = self.processor(
            text=prompts, 
            audio=audios_distorted, 
            return_tensors="pt", 
            padding=True, 
            sampling_rate=16000
        ).to(self.device)
    
        logits_processor = VCDLogitsProcessor(
            self.model,
            inputs_distorted,
            alpha=self.alpha,
            beta=self.beta
        )
        
        return logits_processor

    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> str:
        assert len(texts) == 1, "Currently no batch inference"
        
        # 1. Prepare Original Inputs
        prompts = [self.format_prompt(text) for text in texts]
        inputs_orig = self.processor(
            audio=audios, 
            text=prompts, 
            return_tensors="pt", 
            padding=True, 
            sampling_rate=16000
        ).to(self.device)

        # 2. Prepare Distorted Inputs
        distorted_audios_np = [self.apply_noise(a) for a in audios]
        
        # 3. Initialize VCD Logits Processor
        vcd_processor = self.prepare_logits_processor(prompts, distorted_audios_np)
        
        # 4. Generate
        # The 'inputs_orig' provide the main logic flow (scores in LogitsProcessor)
        # The 'vcd_processor' computes the subtracted logits from the distorted flow
        output_ids = self.model.generate(
            **inputs_orig,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            logits_processor=[vcd_processor]
        )
        
        output_ids = output_ids[:, inputs_orig["input_ids"].size(1):]
        prediction = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Clean up
        del vcd_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction
        }