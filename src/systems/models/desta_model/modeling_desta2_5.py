from typing import Optional, List, Tuple
import numpy as np

from desta.models.modeling_desta25 import *


class DeSTA2_5Model(DeSTA25AudioModel):
    def set_processing_config(self, config):
        self.processing_config = config
        self.audio_placeholder_token_id = config["audio_placeholder_token_id"]
        self.transcription_placeholder_token_id = config["transcription_placeholder_token_id"]
        self.audio_prompt_size = config["audio_prompt_size"]
    
    def _setup_asr(self):
        self.processor = AutoProcessor.from_pretrained(self.config.encoder_model_id, cache_dir=os.getenv("HF_HOME"))

        # VAD
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (self.get_speech_timestamps, _, _, _, _) = utils

    def get_input_embeddings(self):
        return self.llm_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm_model.set_input_embeddings(value)

    def prepare_transcriptions(self, audios: List[np.ndarray], transcriptions: List[Optional[str]]) -> List[str]:
        if getattr(self, "vad_model", None) is None:
            self._setup_asr()
        assert len(audios) == len(transcriptions)  # sanity check
        new_transcriptions = [" "] * len(transcriptions)

        asr_features = []
        asr_indices = []
        for i, (feature, trans) in enumerate(zip(audios, transcriptions)):
            # Transcription is known in advance
            if trans is not None:
                new_transcriptions[i] = trans
                continue

            # Run VAD detect if there is speech in the audio
            is_speech = self.get_speech_timestamps(feature, self.vad_model)
            if is_speech and trans is None:
                asr_features.append(feature)
                asr_indices.append(i)
        
        # Run ASR
        if asr_features:
            inputs = self.processor(
                asr_features,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            transcriptions = self.perception.whisper.generate(
                **inputs,
                max_new_tokens=128,
            )
            transcriptions = self.processor.batch_decode(
                transcriptions,
                skip_special_tokens=True,
            )

            for i, transcription in zip(asr_indices, transcriptions):
                new_transcriptions[i] = transcription.strip()

        return new_transcriptions
    
    def get_audio_features(self, input_features, feature_attention_mask):
        audio_embeds, _ = self.perception(input_features=input_features)
        return audio_embeds.reshape(-1, audio_embeds.shape[-1])  # (n_audio_placeholders, d_input)
    
    def get_transcription_features(self, input_ids, feature_attention_mask):
        transcription_embeds = self.llm_model.model.embed_tokens(input_ids)
        # note that attention_mask is long type, cast it to bool
        return transcription_embeds[feature_attention_mask.bool()]  # (n_transcription_placeholders, d_input)
    
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        audio_features: torch.FloatTensor = None,
        transcription_features: torch.FloatTensor = None,
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_audio_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.audio_placeholder_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_audio_mask = special_audio_mask.all(-1)
            special_transcription_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.transcription_placeholder_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_transcription_mask = special_transcription_mask.all(-1)
        else:
            special_audio_mask = input_ids == self.audio_placeholder_token_id
            special_transcription_mask = input_ids == self.transcription_placeholder_token_id
        
        n_audio_placeholders = special_audio_mask.sum()
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if audio_features is not None and inputs_embeds[special_audio_mask].numel() != audio_features.numel():
            raise ValueError(
                f"Audio features and audio tokens do not match: tokens: {n_audio_placeholders}, features {audio_features.shape[0]}"
            )

        n_transcription_placeholders = special_transcription_mask.sum()
        special_transcription_mask = special_transcription_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if transcription_features is not None and inputs_embeds[special_transcription_mask].numel() != transcription_features.numel():
            raise ValueError(
                f"Transcription features and transcription tokens do not match: tokens: {n_transcription_placeholders}, features {transcription_features.shape[0]}"
            )
        
        # print(n_audio_placeholders, n_transcription_placeholders)

        return special_audio_mask, special_transcription_mask
    
    def _prepare_llm_input_embeds(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        transcription_input_ids: Optional[torch.LongTensor] = None,
        transcription_attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # 1. Extract the input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)

        # 2. Merge text , audios , and transcriptions
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if transcription_input_ids is not None:
            transcription_features = self.get_transcription_features(
                transcription_input_ids,
                feature_attention_mask=transcription_attention_mask
            )
            transcription_features = transcription_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, transcription_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(transcription_mask, transcription_features)

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.BoolTensor] = None,
        transcription_input_ids: Optional[torch.LongTensor] = None,
        transcription_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        inputs_embeds = self._prepare_llm_input_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            transcription_input_ids=transcription_input_ids,
            transcription_attention_mask=transcription_attention_mask
        )

        return self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        transcription_input_ids: Optional[torch.LongTensor] = None,
        transcription_attention_mask: Optional[torch.LongTensor] = None,
        temperature: float=0.7,
        top_p: float=0.9,
        do_sample: bool=True,
        max_new_tokens: int=512,
        **kwargs
    ):
        if do_sample is False:
            top_p = None
            temperature = None

        inputs_embeds = self._prepare_llm_input_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            transcription_input_ids=transcription_input_ids,
            transcription_attention_mask=transcription_attention_mask
        )
        
        return self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs
        )
