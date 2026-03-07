import torch
from src.systems.generation.logits_process import AADLogitsProcessor
import numpy as np
from .audio_flamingo_3 import AudioFlamingo3System

class AADSystem(AudioFlamingo3System):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = self.model_config.get("aad_alpha", 0.5)
        self.threshold = self.model_config.get("aad_threshold", -1)

    def prepare_logits_processor(self, texts) -> AADLogitsProcessor:
        # Prepare Negative Inputs (Text Only - Remove Audio from conversation)
        neg_conversation = self.format_conversation( # No Audio Input
            texts[0]
        )
        neg_inputs = self.processor.apply_chat_template(
            neg_conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.device)
        # print(neg_inputs)
        # print(self.processor.batch_decode(neg_inputs["input_ids"]))

        return AADLogitsProcessor(
            model=self.model,
            inputs_without_audio=neg_inputs,
            alpha=self.alpha,
            threshold=self.threshold
        )

    @torch.inference_mode()
    def inference(self, audios: list, texts: list[str], ids: list[str], max_new_tokens: int = 512) -> dict:
        assert len(texts) == 1, "Batch size 1 recommended for AAD"

        # 1. Prepare Positive Inputs (With Audio)
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
        # print(self.processor.batch_decode(inputs["input_ids"]))

        # 2. Prepare Logits Processor (Handles Negative Stream internally)
        aad_processor = self.prepare_logits_processor(texts)
        
        # 3. Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=[aad_processor] # Inject AAD
        )

        # 4. Decode
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up
        step_all, step_applied = aad_processor.call_cnt, aad_processor.apply_cnt
        del aad_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction,
            "steps": (step_all, step_applied)
        }