import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import LogitsProcessor, DynamicCache
from copy import deepcopy


class ContrastiveLogitsProcessor(LogitsProcessor):
    """
    Logits Processor for Contrastive Decoding.
    
    Args:
        model: The language model.
        embeds_neg: Initial embeddings of the negative input sequence.
        atts_neg: Attention mask corresponding to the negative input.
        alpha: Scaling factor for the contrastive decoding term.
    """
    def __init__(
        self,
        model,
        inputs_neg,
        alpha: float=1.0,
    ):
        self.model = model
        self.alpha = alpha
        
        # Negative Context State
        self.inputs_neg = inputs_neg
        self.atts_neg = inputs_neg["attention_mask"]
        self.past_key_values_neg: DynamicCache = None
        
        self.first_call = True
    
    def __call__(self, input_ids, scores):
        """
        Implements: yt ~ softmax[(1 + α)logit(yt|c,x,y<t) - αlogit(yt|x,y<t)]
        
        Args:
            input_ids: Current sequence of tokens.
            scores: Logits from the main generation path (with audio context).
        """        
        with torch.no_grad():
            if self.first_call:
                self.first_call = False

                outputs_neg = self.model(
                    **self.inputs_neg,
                    use_cache=True,
                    return_dict=True
                )
                self.past_key_values_neg = outputs_neg.past_key_values

            else:
                # Subsequent steps: process only the last generated token
                # We assume the last token in 'input_ids' is the one we need to append to neg stream
                # (The token that was selected in the previous step)
                last_token = input_ids[:, -1:]

                # Extend the attention mask
                new_attention = torch.ones_like(last_token, dtype=self.atts_neg.dtype)
                self.atts_neg = torch.cat([self.atts_neg, new_attention], dim=1)

                # handle cache_position manually (Qwen can not automatically handle our use case)
                past_seen_tokens = self.past_key_values_neg.get_seq_length()
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + 1, device=last_token.device)
                
                outputs_neg = self.model(
                    input_ids=last_token,
                    attention_mask=self.atts_neg,
                    past_key_values=self.past_key_values_neg,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True
                )
                self.past_key_values_neg = outputs_neg.past_key_values
            
            logits_neg = outputs_neg.logits[:, -1, :] # (B, V)

        # Apply AAD Contrastive Formula
        # Formula: (1 + alpha) * logits_pos - alpha * logits_neg
        modified_logits = (1 + self.alpha) * scores - self.alpha * logits_neg

        return modified_logits


class AADLogitsProcessor(ContrastiveLogitsProcessor):
    def __init__(
        self,
        model,
        inputs_without_audio,
        alpha=0.5,
        threshold=-1,
    ):
        """
        Logits Processor for Audio-Aware Decoding (AAD).
        
        Args:
            model: The language model.
            inputs_without_audio: Initial inputs of the input sequence with audio effectively silenced/removed.
            alpha: Scaling factor for the contrastive decoding term.
            threshold: Entropy threshold for selective application (efficiency mechanism).
        """
        super().__init__(
            model=model,
            inputs_neg=inputs_without_audio,
            alpha=alpha,
        )
        
        self.threshold = threshold

        self.apply_cnt = 0
        self.call_cnt = 0

    def __call__(self, input_ids, scores):
        self.call_cnt += 1
        modified_logits = super().__call__(input_ids, scores)

        # Selective Application (Entropy Thresholding)
        apply_mask = self._apply_algo(scores)
        self.apply_cnt += torch.sum(apply_mask.float()).item()
        
        # Expand mask to match vocab size for broadcasting
        apply_mask = apply_mask.unsqueeze(1).expand(-1, scores.size(1))
        
        # Apply modified logits only where entropy is high enough
        modified_logits = torch.where(apply_mask, modified_logits, scores)

        return modified_logits
    
    def _apply_algo(self, scores) -> torch.BoolTensor:
        """
        Calculates entropy of the distribution to decide if AAD should be applied.
        High entropy -> Uncertain generation -> Apply AAD.
        """
        probs = F.softmax(scores, dim=-1)
        # Add small epsilon to avoid log(0)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        return entropy > self.threshold


class VCDLogitsProcessor(ContrastiveLogitsProcessor):
    def __init__(
        self,
        model,
        inputs_distorted,
        alpha=1.0,
        beta=0.1
    ):
        """
        Implements Visual Contrastive Decoding (adapted for Audio).
        
        Args:
            model: The language model
            inputs_distorted: Initial inputs for the distorted (noisy) audio input
            alpha: Hyperparameter for contrastive penalty (default 1.0 from )
            beta: Hyperparameter for adaptive plausibility constraint (default 0.1 from )
        """
        super().__init__(
            model=model,
            inputs_neg=inputs_distorted,
            alpha=alpha,
        )

        self.beta = beta

    def __call__(self, input_ids, scores):
        modified_logits = super().__call__(input_ids, scores)

        # --- 1. Adaptive Plausibility Constraint  ---
        # Calculate P(y|v, x) - Original Probabilities
        probs_original = F.softmax(scores, dim=-1)
        
        # Determine V_head: tokens with prob >= beta * max_prob
        max_probs = probs_original.max(dim=-1, keepdim=True).values
        # Create a mask where valid tokens are True, invalid are False
        plausibility_mask = probs_original >= (self.beta * max_probs)
        
        # --- 2. Apply Constraint ---
        # For tokens NOT in V_head, set logits to -inf (effectively prob 0)
        # We use a very large negative number to represent -inf
        modified_logits = torch.where(plausibility_mask, modified_logits, torch.tensor(-float('inf')).to(modified_logits.device))

        return modified_logits


class AMTILogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        model,
        inputs,
        negative_prompt_tokens,
        omega=1.2,
        tau=0.1,
    ):
        """
        Args:
            model: The language model
            embeds: Initial embeddings
            atts_mask: Attention mask
            negative_prompt_embeds: The embedding of negative prompt for unconditional decoding
            omega: Scaling factor for context-aware decoding
            tau: Threshold for entropy
        """
        self.model = model
        self.negative_prompt_tokens = negative_prompt_tokens
        self.omega, self.alpha = omega, omega - 1
        self.tau = tau
        
        # Context State
        self.suffix = None
        self.inputs_orig = inputs
        self.atts_orig = inputs["attention_mask"]
        with torch.no_grad():
            self.past_key_values_orig = self.model(
                **self.inputs_orig,
                use_cache=True,
                return_dict=True
            ).past_key_values

        self.first_call = True

    def cal_entropy(self, logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = (probs + 1e-9).log2()
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def __call__(self, input_ids, scores):
        if self.first_call:
            self.first_call = False
            assert self.suffix is None
        else:
            last_token = input_ids[:, -1:]
            self.suffix = last_token.clone() if self.suffix is None else torch.cat([self.suffix, last_token], dim=1)
        
        entropy = self.cal_entropy(scores)
        if entropy <= self.tau:
            return scores
        
        # AMTI
        with torch.no_grad():
            bs = input_ids.shape[0]
            assert bs == 1, "currently does not support batch inference"
            suffix_neg = self.negative_prompt_tokens.repeat(bs, 1) if self.suffix is None else \
                torch.cat([self.suffix, self.negative_prompt_tokens.repeat(bs, 1)], dim=1)
            
            # Extend the attention mask
            new_attention = torch.ones_like(suffix_neg, dtype=self.atts_orig.dtype)
            atts_neg = torch.cat([self.atts_orig, new_attention], dim=1)

            # handle cache_position manually (Qwen can not automatically handle our use case)
            past_seen_tokens = self.past_key_values_orig.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + suffix_neg.shape[-1],
                device=suffix_neg.device
            )

            # print(self.past_key_values_orig[0][0].shape)
            outputs_neg = self.model(  # equivalent to forward pass [inputs_orig.input_ids][suffix_neg] together
                input_ids=suffix_neg,
                attention_mask=atts_neg,
                past_key_values=deepcopy(self.past_key_values_orig),  # kv cache updates in-place during forward()
                cache_position=cache_position,
                use_cache=True,
                return_dict=True
            )
            # print(suffix_neg.shape, atts_neg.shape, outputs_neg.past_key_values[0][0].shape)

            logits_neg = outputs_neg.logits[:, -1, :] # (B, V)

            # matching
            # tmp = deepcopy(self.inputs_orig)
            # tmp["input_ids"] = torch.cat([tmp["input_ids"], suffix_neg], dim=1)
            # tmp["attention_mask"] = atts_neg
            # match_neg = self.model(**tmp).logits[:, -1, :]
            # print(torch.mean((F.softmax(match_neg, dim=-1) - F.softmax(logits_neg, dim=-1)) ** 2))
            # assert 1 == 2

        # Apply Contrastive Formula
        modified_logits = (1 + self.alpha) * scores - self.alpha * logits_neg

        return modified_logits

class DoLALogitsProcessor(LogitsProcessor):
    def __init__(
        self, 
        system_ref, 
        candidate_layers: list[int],
        final_norm: nn.Module,
        lm_head: nn.Module,
        num_layers: int,
        jsd_threshold: float = 0.1, 
        mode: str = "dynamic",
    ):
        """
        Args:
            system: Reference to the DoLASystem to access captured hidden states.
            candidate_layers: List of layer indices to consider as 'premature' layers.
            final_norm: The final normalization layer applied before the output projection.
            lm_head: The language modeling head used to calculate final logits.
            num_layers: The total number of layers in the model.
            jsd_threshold: (Optional) Threshold for applying DoLa (similar to AAD's entropy threshold).
            mode: 'dynamic' (select layer with max JSD) or 'fixed' (use first candidate).
        """
        self.system = system_ref
        assert hasattr(self.system, 'captured_states'), "A dictionary named captured_states should be implemented in DoLASystem to capture hidden states from premature layers."
        self.candidate_layers = candidate_layers
        assert len(self.candidate_layers) != 0, "The number of candidate layers cannot be zero."
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.num_layers = num_layers
        # Include every second layer as a candidate, starting from the middle of the model.
        self.jsd_threshold = jsd_threshold
        self.mode = mode   

    def _get_jsd(self, p: torch.Tensor, q: torch.Tensor):
        """
        Calculate Jensen-Shannon Divergence between two probability distributions.
        p, q: (B, V) probabilities
        """
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(m.log(), p, reduction='none').sum(dim=-1) + 
                      F.kl_div(m.log(), q, reduction='none').sum(dim=-1))

    def __call__(self, input_ids, scores):
        # print("======================================")
        # 1. Retrieve captured hidden states & Compute premature logits
        premature_logits_candidates = []
        for layer_idx in self.candidate_layers:
            assert layer_idx in self.system.captured_states, f"hidden states of layer {layer_idx} are not properly captured."
            hidden_state = self.system.captured_states[layer_idx]
            with torch.no_grad():
                if self.num_layers != layer_idx+1:
                    normalized_hidden = self.final_norm(hidden_state)
                else:
                    normalized_hidden = hidden_state
                logits = self.lm_head(normalized_hidden).squeeze(1)
                premature_logits_candidates.append((layer_idx, logits))

        assert len(premature_logits_candidates) != 0

        # 2. Convert to probabilities for JSD calculation
        probs_N = F.softmax(scores, dim=-1)
        
        # 3. Select best premature layer
        best_premature_logits = None
        
        if self.mode == "dynamic":
            max_jsd = -1.0
            for layer_idx, p_logits in premature_logits_candidates:
                prem_probs = F.softmax(p_logits, dim=-1)
                jsd = self._get_jsd(probs_N, prem_probs)
                current_jsd = jsd.mean().item() # Simplify for batch
                if current_jsd >= max_jsd:
                    max_jsd = current_jsd
                    best_premature_logits = p_logits
        else:
            best_premature_logits = premature_logits_candidates[0][1]

        assert best_premature_logits is not None

        probs_M = F.softmax(best_premature_logits, dim=-1)

        # 4. Adaptive Plausibility Constraint (APC)
        top_prob, _ = probs_N.max(dim=-1, keepdim=True)
        mask = probs_N >= (self.jsd_threshold * top_prob)
        
        # 5. Contrastive Logic 
        log_diff = torch.log(probs_N) - torch.log(probs_M)

        processed_logits = torch.where(mask, log_diff, torch.tensor(float('-inf')).to(scores.device))
        
        return processed_logits
