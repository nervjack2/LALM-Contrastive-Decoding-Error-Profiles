import torch
import torch.nn.functional as F
from src.systems.generation.logits_process import VCDLogitsProcessor
import numpy as np
from .audio_flamingo_3 import AudioFlamingo3System

class ACDSystem(AudioFlamingo3System):
    def __init__(self, config):
        super().__init__(config)
        acd_config = self.model_config.get("vcd", {})
        self.alpha = acd_config.get("alpha", 1.0)
        self.beta = acd_config.get("beta", 0.1)
        self.noise_gamma = self.config.get("vcd", {}).get("gamma", 0.1) 
        self.noise_steps = self.config.get("vcd", {}).get("steps", 100)

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

    def prepare_logits_processor(self, texts, audios) -> VCDLogitsProcessor:
        distorted_audios = self.apply_noise(audios[0]) # Assuming batch size 1
        dist_conversations = self.format_conversation(texts[0], audio_path_or_array=distorted_audios)

        dist_inputs = self.processor.apply_chat_template(
            dist_conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.device)

        return VCDLogitsProcessor(
            model=self.model,
            inputs_distorted=dist_inputs,
            alpha=self.alpha,
            beta=self.beta
        )

    @torch.inference_mode()
    def inference(self, audios: list, texts: list[str], ids: list[str], max_new_tokens: int = 512) -> dict:
        assert len(texts) == 1, "Batch size 1 recommended for ACD"

        # 1. Prepare Positive Inputs (Original Audio)
        conversation = self.format_conversation(
            texts[0], 
            ids[0]
        )
        
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.device)

        # 2. Prepare Logits Processor
        acd_processor = self.prepare_logits_processor(texts, audios)

        # 3. Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=False,
            logits_processor=[acd_processor] # Inject ACD
        )

        # 4. Decode
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up
        del acd_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction
        }