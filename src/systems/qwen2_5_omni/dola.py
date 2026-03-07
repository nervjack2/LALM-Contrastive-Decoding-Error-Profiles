import torch
import torch.nn.functional as F
from src.systems.generation.logits_process import DoLALogitsProcessor
from .qwen2_5_omni import Qwen2_5OmniSystem

class DoLASystem(Qwen2_5OmniSystem):
    def __init__(self, config):
        super().__init__(config)
        self.dola_mode = self.model_config.get("mode", "dynamic")
        self.captured_states = {}
        self.hooks = []
        self.num_layers = len(self.model.model.layers)
        self.candidate_layers = list(range(self.num_layers//2-1, self.num_layers-1, 2))
        for layer_idx in self.candidate_layers:
            assert 0 <= layer_idx < self.num_layers, f"layer {layer_idx} out of bound."
        self.final_norm = self.model.model.norm
        self.lm_head = self.model.lm_head

    def _forward_hook_fn(self, layer_idx):
        def hook(module, input, output):
            hidden_states = output[0]
            self.captured_states[layer_idx] = hidden_states[:, -1:, :].detach()
        return hook

    def register_dola_hooks(self):
        """
        Attaches forward hooks to the specific transformer layers we want to contrast.
        Qwen2 structure: model.model.layers[i]
        """
        self.captured_states = {} # Clear cache
        self.hooks = []

        layers = self.model.model.layers
        
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
    def inference(self, audios: list, texts: list, ids: list[str], max_new_tokens: int = 512) -> str:
        assert len(texts) == 1, "Currently no batch inference"
        
        prompts = [self.format_prompt(text) for text in texts]
        inputs = self.processor(audio=audios, text=prompts, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
        
        # 1. Register Hooks before generation
        self.register_dola_hooks()
        
        # 2. Prepare Logits Processor
        dola_processor = self.prepare_logits_processor()
        
        # 3. Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            logits_processor=[dola_processor]
        )
        # 4. Clean up hooks strictly
        self.remove_hooks()
            
        output_ids = output_ids[:, inputs["input_ids"].size(1):]
        prediction = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return {
            "prediction": prediction
        }