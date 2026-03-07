import os
import numpy as np
from typing import Optional, List
from transformers import AutoTokenizer, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature


class Desta2_5Processor(object):
    def __init__(
        self, text_tokenizer_id: str, speech_processor_id: str, processing_config: dict
    ):  
        self.audio_prompt_size = processing_config["audio_prompt_size"]
        self.audio_locator = processing_config["audio_locator"]
        self.audio_placeholder_token = processing_config["audio_placeholder_token"]
        self.transcription_placeholder_token = processing_config["transcription_placeholder_token"]
        self.transcription_locator = "<|TRANS|>"

        self.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_id, cache_dir=os.getenv("HF_HOME"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_tokens([self.audio_locator])

        self.processor = AutoProcessor.from_pretrained(speech_processor_id, cache_dir=os.getenv("HF_HOME"))
        print(speech_processor_id)

        assert len(self.tokenizer.tokenize(self.audio_locator)) == 1, "audio_locator must be a single token"
        assert len(self.tokenizer.tokenize(self.audio_placeholder_token)) == 1, "placeholder_token must be a single token in the tokenizer"
        assert len(self.tokenizer.tokenize(self.transcription_placeholder_token)) == 1, "placeholder_token must be a single token in the tokenizer"

    def __call__(
        self,
        audio: Optional[np.ndarray]=None,
        transcription: Optional[List[str]]=None,
        text: Optional[List[str]]=None,
        **kwargs
    ) -> BatchFeature:
        """
        Args:
            text (`list[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            
            audio (`list[np.ndarray]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array.
            transcription (`list[str]`):
                The transcription correspond to the audio.
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
        
        if not isinstance(text, list):
            text = [text]
        
        # check audio count
        total_audio_count = 0
        for content in text:
            total_audio_count += content.count(self.audio_locator)
        assert total_audio_count == 0 or total_audio_count == len(audio), "audio count does not match (<|AUDIO|>) count"

        if audio is not None:
            assert transcription is not None and len(audio) == len(transcription), "number of audio and transcription mismatched"
            audio_inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            audio_lengths = iter([self.audio_prompt_size] * len(audio))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if transcription is not None:
            # insert transcription locator after audio_locator
            text = [x.replace(self.audio_locator, f"{self.audio_locator}{self.transcription_locator}") for x in text]
            transcription_inputs = self.tokenizer(transcription, add_special_tokens=False, return_tensors="pt", return_attention_mask=True)
            transcription_inputs["transcription_input_ids"] = transcription_inputs.pop(
                "input_ids"
            )  # rename input_ids to prevent conflicts later on
            transcription_inputs["transcription_attention_mask"] = transcription_inputs.pop(
                "attention_mask"
            )  # rename attention_mask to prevent conflicts later on
            transcription_lengths = iter(transcription_inputs["transcription_attention_mask"].sum(dim=-1))
        else:
            transcription_inputs = {}
            transcription_lengths = iter([])

        # insert audio_placeholder_token
        if audio is not None:
            process_text = []
            for sample in text:
                while self.audio_locator in sample:
                    sample = sample.replace(self.audio_locator, "<|audio_placeholder|>" * next(audio_lengths), 1)
                sample = sample.replace("<|audio_placeholder|>", self.audio_placeholder_token)
                process_text.append(sample)
            text = process_text

        # insert transcription_placeholder_token
        if transcription is not None:
            process_text = []
            for sample in text:
                while self.transcription_locator in sample:
                    sample = sample.replace(self.transcription_locator, "<|transcription_placeholder|>" * next(transcription_lengths), 1)
                sample = sample.replace("<|transcription_placeholder|>", self.transcription_placeholder_token)
                process_text.append(sample)
            text = process_text

        texts_inputs = self.tokenizer(text, **kwargs)

        return BatchFeature(
            data={**texts_inputs, **transcription_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def apply_chat_template(self, messages, add_generation_prompt=False) -> list[str]:
        """
        The largest different with original implementation is that transcription is now REQUIRED.
        This largely reduce the complexity since processor know the correct number of placeholders in input ids in advance.
        This trades the convenience of usage for explicit control due to research.

        messages = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions.",
            },
            {
                "role": "user",
                "content": "Hello! this is my audio <|AUDIO|>. Help me transcribe."
                "audios": [
                    "audio": "/path/to/filepath", # path to audio file
                    "text": " ", # provide text, if unknown, use prepare_transcriptions()
                ]
            },
        ]
        """
        if isinstance(messages, list):
            if isinstance(messages[0], dict):
                messages_list = [messages]
                is_batched = False
            else: 
                messages_list = messages
                is_batched = True
        else:
            raise ValueError("messages should be a list of dictionaries or a list of lists.")
        
        prompt = self.tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        # <start_audio><|AUDIO|><end_audio> is a indicator used in the training stage
        # We replace <|AUDIO|> with <start_audio><|AUDIO|><end_audio> here
        prompt = [p.replace(self.audio_locator, f"<start_audio>{self.audio_locator}<end_audio>") for p in prompt]

        return prompt if is_batched else prompt[0]
