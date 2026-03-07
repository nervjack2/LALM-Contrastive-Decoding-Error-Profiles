import numpy as np
import torch
import torch.nn.functional as F
from src.systems.generation.logits_process import LIMLogitsProcessor
from copy import deepcopy
from .qwen2_5_omni import Qwen2_5OmniSystem

class LIMSystem(Qwen2_5OmniSystem):
    def __init__(self, config):
        super().__init__(config)
        self.omega = self.model_config["lim"].get("omega", 1.2)
        self.tau = self.model_config["lim"].get("tau", 0.1)
        self.negative_prompt = self.model_config["lim"].get("negative_prompt", "Output Error")
        self.negative_prompt_tokens = self.processor(
            text=[self.negative_prompt],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(self.device)["input_ids"]
    
    def prepare_logits_processor(self, inputs) -> LIMLogitsProcessor:
        logits_processor = LIMLogitsProcessor(
            model=self.model, 
            inputs=inputs,
            negative_prompt_tokens=self.negative_prompt_tokens, 
            omega=self.omega, 
            tau=self.tau
        )
        return logits_processor

    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> str:
        assert len(texts) == 1, "Currently no batch inference"
        
        prompts = [self.format_prompt(text) for text in texts]

        # Prepare inputs
        inputs = self.processor(audio=audios, text=prompts, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)

        # prepare logits processor
        lim_logits_processor = self.prepare_logits_processor(inputs.copy())

        # text-only generation
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            logits_processor=[lim_logits_processor]
        )
        output_ids = output_ids[:, inputs["input_ids"].size(1):]

        prediction = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # clean up
        del lim_logits_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction
        }