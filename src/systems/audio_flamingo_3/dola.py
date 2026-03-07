import torch
from src.systems.generation.logits_process import DoLALogitsProcessor
from .audio_flamingo_3 import AudioFlamingo3System

class DoLASystem(AudioFlamingo3System):
    def __init__(self, config):
        super().__init__(config)
        # DoLa Configuration
        self.dola_mode = self.model_config.get("mode", "dynamic")
        self.captured_states = {}
        self.hooks = []
        self.num_layers = len(self.model.language_model.model.layers)
        self.candidate_layers = list(range(self.num_layers // 2 - 1, self.num_layers - 1, 2))
        for layer_idx in self.candidate_layers:
            assert 0 <= layer_idx < self.num_layers, f"layer {layer_idx} out of bound."
        self.final_norm = self.model.language_model.model.norm
        self.lm_head = self.model.language_model.lm_head
        # print(self.num_layers, self.candidate_layers)
        # print(self.model.language_model)

    def _forward_hook_fn(self, layer_idx):
        """
        Hook function to capture the output hidden state of a specific layer.
        """
        def hook(module, input, output):
            # print(output, layer_idx, output.shape)
            hidden_states = output
            self.captured_states[layer_idx] = hidden_states[:, -1:, :].detach()
        return hook

    def register_dola_hooks(self):
        """
        Registers forward hooks on the transformer layers.
        """
        self.captured_states = {} 
        self.hooks = []
        
        layers = self.model.language_model.model.layers
        
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
        assert len(texts) == 1, "Batch size 1 recommended for DoLa"

        # 1. Format Inputs (Standard AF3 flow)
        conversation = self.format_conversation(texts[0], ids[0])
        
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.device)
        # print(inputs)

        # 2. Register DoLa Hooks
        self.register_dola_hooks()
        
        # 3. Prepare Logits Processor
        dola_processor = self.prepare_logits_processor()
        
        # 4. Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            logits_processor=[dola_processor]
        )

        self.remove_hooks()

        # 6. Decode
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        del dola_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction,
        }