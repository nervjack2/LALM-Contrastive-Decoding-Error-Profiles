import torch
from src.systems.generation.logits_process import DoLALogitsProcessor
from .desta2_5 import Desta2_5System

class DoLASystem(Desta2_5System):
    def __init__(self, config):
        super().__init__(config)
        # DoLa Configuration
        self.dola_mode = self.model_config.get("mode", "dynamic")
        self.captured_states = {}
        self.hooks = []
        self.num_layers = len(self.model.llm_model.model.layers)
        self.candidate_layers = list(range(self.num_layers // 2 - 1, self.num_layers - 1, 2))
        for layer_idx in self.candidate_layers:
            assert 0 <= layer_idx < self.num_layers, f"layer {layer_idx} out of bound."
        self.final_norm = self.model.llm_model.model.norm
        self.lm_head = self.model.llm_model.lm_head

    def _forward_hook_fn(self, layer_idx):
        """
        Hook function to capture the output hidden state of a specific layer.
        """
        def hook(module, input, output):
            hidden_states = output
            self.captured_states[layer_idx] = hidden_states[:, -1:, :].detach()
        return hook

    def register_dola_hooks(self):
        """
        Registers forward hooks on the transformer layers.
        """
        self.captured_states = {} 
        self.hooks = []
        
        layers = self.model.llm_model.model.layers
        
        for layer_idx in self.candidate_layers:
            handle = layers[layer_idx].register_forward_hook(self._forward_hook_fn(layer_idx))
            self.hooks.append(handle)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.captured_states = {}

    def prepare_logits_processor(self) -> DoLALogitsProcessor:
        return DoLALogitsProcessor(
            system_ref=self,
            candidate_layers=self.candidate_layers,
            final_norm=self.final_norm,
            lm_head=self.lm_head,
            num_layers=self.num_layers,
            mode=self.dola_mode
        )

    @torch.inference_mode()
    def inference(self, audios: list, texts: list[str], ids: list[str], max_new_tokens: int = 512) -> dict:
        assert len(texts) == 1, "Currently no batch inference"

        # prepare inputs
        transcriptions = self.model.prepare_transcriptions(audios, [None])

        prompts = [
            self.format_prompt(audio, transcription, text)
        for (audio, transcription, text) in zip(audios, transcriptions, texts)]

        inputs = self.processor(
            audio=audios, transcription=transcriptions, text=prompts,
            add_special_tokens=False,
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        
        self.register_dola_hooks()
        
        dola_processor = self.prepare_logits_processor()

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=[dola_processor],
        )

        self.remove_hooks()
        prediction = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        del dola_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction
        }