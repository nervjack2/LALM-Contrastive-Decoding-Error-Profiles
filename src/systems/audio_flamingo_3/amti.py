import torch
from src.systems.generation.logits_process import AMTILogitsProcessor
from .audio_flamingo_3 import AudioFlamingo3System

class AMTISystem(AudioFlamingo3System):
    def __init__(self, config):
        super().__init__(config)
        self.amti_config = self.model_config.get("amti", {})
        self.omega = self.amti_config.get("omega", 2.0)
        self.tau = self.amti_config.get("tau", 1.0)
        self.negative_prompt = self.amti_config.get("negative_prompt", "Output Error")
        self.negative_prompt_tokens = self.processor(
            text=[self.negative_prompt],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(self.device)["input_ids"]

    def prepare_logits_processor(self, inputs) -> AMTILogitsProcessor:
        return AMTILogitsProcessor(
            model=self.model,
            inputs=inputs,
            negative_prompt_tokens=self.negative_prompt_tokens, 
            omega=self.omega, 
            tau=self.tau
        )

    @torch.inference_mode()
    def inference(self, audios: list, texts: list[str], ids: list[str], max_new_tokens: int = 512) -> dict:
        assert len(texts) == 1, "Batch size 1 recommended for AMTI"

        # 1. Prepare Inputs (With Audio) - This is the standard path
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
        amti_processor = self.prepare_logits_processor(inputs)

        # 3. Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=[amti_processor] # Inject AMTI
        )

        # 4. Decode
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up
        del amti_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction
        }