from transformers import AutoProcessor
from ..base import System
from transformers import AudioFlamingo3ForConditionalGeneration

class AudioFlamingo3System(System):
    """
    Audio Flamingo 3 Wrapper using Hugging Face Transformers
    """
    def __init__(self, config):
        self.config = config
        model_id = "nvidia/audio-flamingo-3-hf"
        self.model_config = self.config["model_config"]
        self.device = "cuda"

        # Load Processor
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Load Model
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto", 
            device_map="auto"
        )
        self.model.eval()

    def format_conversation(self, text: str, audio_path_or_array=None) -> list:
        content = [{"type": "text", "text": text}]
        
        if audio_path_or_array is not None:
            content.append({"type": "audio", "path": audio_path_or_array})

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def inference(self, audios: list, texts: list[str], ids: list[str], max_new_tokens: int = 512) -> dict:
        assert len(texts) == 1, "Batch inference not implemented for simplicity"
        
        conversation = [self.format_conversation(text, id_) for text, id_ in zip(texts, ids)]
        
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.device)

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {"prediction": prediction}