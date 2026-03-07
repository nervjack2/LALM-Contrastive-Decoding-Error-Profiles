import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from copy import deepcopy

from ..base import System


class Beam(object):
    def __init__(self, data: dict, system_name: str):
        self.system_name = system_name
        self.generated = []
        self.data = data

    def add_token(self, new_token) -> None:
        if isinstance(new_token, int):
            new_token = torch.LongTensor([new_token]).to(self.data["input_ids"].device)
        self.generated.append(new_token)
        self.data["input_ids"] = torch.cat([self.data["input_ids"], new_token.unsqueeze(-1)], dim=-1)

        # Update attention_mask if present
        if "attention_mask" in self.data:
            new_mask = torch.ones((self.data["attention_mask"].shape[0], 1), device=new_token.device, dtype=self.data["attention_mask"].dtype)
            self.data["attention_mask"] = torch.cat([self.data["attention_mask"], new_mask], dim=-1)

    def copy(self) -> "Beam":
        beam = Beam(deepcopy(self.data), self.system_name)
        beam.generated = deepcopy(self.generated)
        return beam


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
    
    def format_full_prompt(self, question: str, answer: str) -> str:
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "x"},  # we only need a placeholder here, will load it manually in inference() 
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, tokenize=False)
        # print(prompt)
        
        return prompt[:-1]  # get rid of "\n" at the end
    
    def get_beam(self, audio_input, text_input) -> Beam:
        audios = [audio_input]
        prompts = [self.format_prompt(text_input)]
        inputs = self.processor(
            audio=audios,
            text=prompts,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000
        ).to(self.device)

        return Beam(inputs, system_name=self.config["system_name"])

    @staticmethod
    def find_audio_positions(input_ids):
        audio_start_token_id, audio_end_token_id = 151647, 151648
        sts, eds = [], []
        for b in range(input_ids.shape[0]):
            start_pos = (input_ids[b] == audio_start_token_id).nonzero(as_tuple=True)[0]
            end_pos = (input_ids[b] == audio_end_token_id).nonzero(as_tuple=True)[0]
            if len(start_pos) == 0:
                return None, None
            else:
                sts.append(start_pos.item())
                eds.append(end_pos.item())
        return sts, eds

    def get_input_length(self, audio: np.ndarray, text: str) -> int:
        prompts = [self.format_prompt(text)]
        inputs = self.processor(audio=[audio], text=prompts, return_tensors="pt", padding=True, sampling_rate=16000)
        return inputs.input_ids.size(1)
        
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
    
    @torch.inference_mode()
    def get_perplexity(self, audios: list[np.ndarray], texts: list[str]) -> list[float]:
        assert audios is None or len(audios) == len(texts)
        prompts = [self.format_prompt(text) for text in texts]
        
        # Prepare inputs
        inputs = self.processor(audio=audios, text=prompts, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
        input_ids = inputs["input_ids"]
        B, L = input_ids.shape

        # Create labels initialized as a copy
        labels = input_ids.clone()

        # Mask out positions before audio end token
        audio_end_token_id = 151648  # hardcoded!
        for b in range(B):
            # find the index of the first occurrence of the audio_end_token_id
            pos = (input_ids[b] == audio_end_token_id).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                cutoff = pos.item()  # keep everything AFTER this token
                labels[b, :cutoff + 1] = -100  # mask out audio + audio_end itself
            else:
                raise NotImplementedError
        # print(prompts[0])
        # print(inputs.input_ids)
        # print(labels)

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits.to(torch.float32)  # (B, L, V)

        # Shift for causal LM loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute cross entropy per token (no reduction yet)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        # flatten for CE
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )  # (B*(L-1),)

        # Reshape back to (B, L-1)
        loss = loss.view(B, L - 1)

        # Mask out ignored tokens (-100)
        mask = (shift_labels != -100)
        loss = loss * mask

        # Average over valid tokens per sequence
        seq_loss = loss.sum(dim=1) / mask.sum(dim=1)

        # Perplexity = exp(loss)
        ppl = torch.exp(seq_loss).tolist()

        return ppl

    @torch.inference_mode()
    def get_answer_prob(self, audios: list[np.ndarray], texts: list[str], answers: list[str]) -> list[float]:
        assert len(texts) == len(answers)
        assert audios is None or len(audios) == len(texts)
        full_prompts = [self.format_full_prompt(text, answer) for (text, answer) in zip(texts, answers)]
        prefix_prompts = [self.format_prompt(text) for text in texts]
        
        # Prepare inputs
        inputs = self.processor(audio=audios, text=full_prompts, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
        input_ids = inputs["input_ids"]
        B, L = input_ids.shape

        # Create labels initialized as a copy
        labels = input_ids.clone()

        # Determine prefix lengths for each sample
        prefix = self.processor(audio=audios, text=prefix_prompts, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
        prefix_ids = prefix["input_ids"]
        full_lengths = (input_ids != self.processor.tokenizer.pad_token_id).sum(dim=1)  # (B,)
        prefix_lengths = (prefix_ids != self.processor.tokenizer.pad_token_id).sum(dim=1)  # (B,)
        
        for b in range(B):
            labels[b, :-(full_lengths[b] - prefix_lengths[b])] = -100  # -100 tells loss to ignore non-answer part
        # print(labels)

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits.to(torch.float32)  # (B, L, V)

        # Shift for causal LM loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute cross entropy per token (no reduction yet)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        # flatten for CE
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )  # (B*(L-1),)

        # Reshape back to (B, L-1)
        loss = loss.view(B, L - 1)

        # Mask out ignored tokens (-100)
        mask = (shift_labels != -100)
        loss = loss * mask

        seq_loss = loss.sum(dim=1)
        answer_probs = torch.exp(-seq_loss).tolist()

        return answer_probs


class RepeatSystem(Qwen2_5OmniSystem):
    def __init__(self, config):
        super().__init__(config)
        self.use_emb = self.model_config["repeat"].get("use_emb", False)
        self.n_repeat = self.model_config["repeat"].get("n_repeat", 1)

    def repeat_audios(self, audios: list[np.ndarray]) -> list[np.ndarray]:
        if self.use_emb:
            res = []
            for x in audios:
                res.extend([x] * self.n_repeat)
        else:
            res = [np.concatenate([x] * self.n_repeat, axis=0) for x in audios]
        
        return res
    
    def get_beam(self, audio_input, text_input) -> Beam:
        audios = self.repeat_audios([audio_input])
        prompts = [self.format_prompt(text_input)]
        inputs = self.processor(
            audio=audios,
            text=prompts,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000
        ).to(self.device)

        return Beam(inputs, system_name=self.config["system_name"])
    
    def format_prompt(self, text: str) -> str:
        if self.use_emb:
            tmp = [{"type": "audio", "audio_url": "x"}] * self.n_repeat
        else:
            tmp = [{"type": "audio", "audio_url": "x"}]
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    *tmp,
                    {"type": "text", "text": text},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        return prompt
    
    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> str:
        assert len(texts) == 1, "Currently no batch inference"
        prompts = [self.format_prompt(text) for text in texts]
        audios = self.repeat_audios(audios)

        # Prepare inputs
        inputs = self.processor(audio=audios, text=prompts, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
        
        # text-only generation
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        output_ids = output_ids[:, inputs["input_ids"].size(1):]

        prediction = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return {
            "prediction": prediction
        }


class NegativeSystem(Qwen2_5OmniSystem):
    def __init__(self, config):
        super().__init__(config)
        self.use_silence = self.model_config["negative"].get("use_silence", False)

    def process_audios(self, audios: list[np.ndarray]) -> list[np.ndarray]:
        if self.use_silence:
            res = []
            for x in audios:
                res.append(np.zeros_like(x))
        else:
            res = None
        return res
    
    def get_beam(self, audio_input, text_input) -> Beam:
        audios = self.process_audios([audio_input])
        prompts = [self.format_prompt(text_input)]
        inputs = self.processor(
            audio=audios,
            text=prompts,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000
        ).to(self.device)

        return Beam(inputs, system_name=self.config["system_name"])
    
    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], max_new_tokens: int = 512) -> str:
        assert len(texts) == 1, "Currently no batch inference"
        prompts = [self.format_prompt(text) for text in texts]
        audios = self.process_audios(audios)

        # Prepare inputs
        inputs = self.processor(audio=audios, text=prompts, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
        
        # text-only generation
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        output_ids = output_ids[:, inputs["input_ids"].size(1):]

        prediction = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return {
            "prediction": prediction
        }
