import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from src.systems.models import DeSTA2_5Model, Desta2_5Processor
from ..base import System


class Desta2_5System(System):
    """
    Refactored Desta-2.5 to align more with HF generation interface
    """
    def __init__(self, config):
        model_id = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"
        self.config = config
        self.model_config = self.config["model_config"]
        self.device = "cuda"

        # Load
        self.model = DeSTA2_5Model.from_pretrained(model_id)
        self.model.to(self.device)

        processing_config = {
            "audio_prompt_size": self.model.config.prompt_size,
            "audio_locator": self.model.config.audio_locator,
            "audio_placeholder_token": "<|reserved_special_token_87|>",
            "transcription_placeholder_token": "<|reserved_special_token_88|>",
            "audio_placeholder_token_id": 128096,
            "transcription_placeholder_token_id": 128097,
        }
        self.processor = Desta2_5Processor(
            text_tokenizer_id=self.model.config.llm_model_id,
            speech_processor_id=self.model.config.encoder_model_id,
            processing_config=processing_config,
        )
        self.tokenizer = self.processor.tokenizer

        self.model.set_processing_config(processing_config)

    def format_prompt(self, audio, transcription, text: str):
        conversation = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions."
            }
        ]
        if audio is None:
            conversation.append({
                "role": "user",
                "content": text
            })
        else:
            conversation.append({
                "role": "user",
                "content": f"<|AUDIO|>\n{text}",
                "audios": [{
                    "audio": audio,
                    "text": transcription
                }]
            })
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print(prompt)
        
        return prompt

    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> str:
        assert len(texts) == 1, "Currently no batch inference"
        
        # prepare inputs
        transcriptions = self.model.prepare_transcriptions(audios, [None])
        # print(transcriptions)

        prompts = [
            self.format_prompt(audio, transcription, text)
        for (audio, transcription, text) in zip(audios, transcriptions, texts)]

        inputs = self.processor(
            audio=audios, transcription=transcriptions, text=prompts,
            add_special_tokens=False,
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # text-only generation
        # Note that since we unify all modalities by using input_embeds implicity,
        # output_ids does not contain input_ids as usual
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # output_ids = output_ids[:, inputs["input_ids"].size(1):]

        prediction = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        return {
            "prediction": prediction
        }


class OfficialSystem(System):
    """
    Refer to official implementation
    """

    def __init__(self, config):
        from desta import DeSTA25AudioModel
        model_id = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"
        self.config = config
        self.model_config = self.config["model_config"]
        self.device = "cuda"

        # Load the model from Hugging Face
        self.model = DeSTA25AudioModel.from_pretrained(model_id)
        self.model.to(self.device)

    def format_prompt(self, audio: np.ndarray, text: str) -> dict:
        prompt = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions."
            },
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{text}",
                "audios": [{
                    "audio": audio,
                    "text": None
                }]
            }
        ]
        
        return prompt
    
    def get_beam(self, audio_input, text_input):
        raise NotImplementedError

    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> str:
        assert len(texts) == 1, "Currently no batch inference"
        messages = self.format_prompt(audios[0], texts[0])

        # Prepare inputs
        inputs = messages
        
        # text-only generation
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        prediction = outputs.text[0]

        return {
            "prediction": prediction
        }
