import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from copy import deepcopy

from ..base import System


class Qwen2_5OmniSystem(System):
    """
    Use Qwen/Qwen2.5-Omni for text and audio inference (HF version)
    """

    def __init__(self, config):
        model_id = "Qwen/Qwen2.5-Omni-7B"
        self.config = config
        self.model_config = self.config["model_config"]
        self.device = "cuda"

        # Load processor
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

        # Load model
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model.eval()

    def format_prompt(self, text: str, audio_exist: bool = True) -> str:
        user_content = []

        if audio_exist:
            user_content.append({"type": "audio", "audio_url": "x"})

        user_content.append({"type": "text", "text": text})

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        return prompt

    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> str:
        assert len(texts) == 1, "Currently no batch inference"
        prompts = [self.format_prompt(text) for text in texts]
        # print(prompts[0])

        # Prepare inputs
        inputs = self.processor(audio=audios, text=prompts, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
        
        # text-only generation
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        output_ids = output_ids[:, inputs["input_ids"].size(1):]

        prediction = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return {
            "prediction": prediction
        }