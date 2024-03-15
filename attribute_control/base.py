from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import torch
from torch import nn
from jaxtyping import Float


@dataclass
class PromptEmbedding:
    prompt: str
    tokenwise_embeddings: Dict[str, Float[torch.Tensor, 'n d']]
    tokenwise_embedding_spans: Dict[str, List[Tuple[int, int]]]
    pooled_embeddings: Dict[str, Float[torch.Tensor, 'd']] = field(default_factory=dict)

    @staticmethod
    def merge(*embs: PromptEmbedding) -> PromptEmbedding:
        emb_joined = None
        for emb in embs:
            if emb_joined is None:
                emb_joined = PromptEmbedding(
                    prompt=emb.prompt,
                    tokenwise_embeddings={ k: v for k, v in emb.tokenwise_embeddings.items() },
                    tokenwise_embedding_spans={ k: v for k, v in emb.tokenwise_embedding_spans.items() },
                    pooled_embeddings={ k: v for k, v in emb.pooled_embeddings.items() },
                )
            else:
                assert emb.prompt == emb_joined.prompt
                emb_joined.tokenwise_embeddings = emb_joined.tokenwise_embeddings | emb.tokenwise_embeddings
                emb_joined.tokenwise_embedding_spans = emb_joined.tokenwise_embedding_spans | emb.tokenwise_embedding_spans
                emb_joined.pooled_embeddings = emb_joined.pooled_embeddings | emb.pooled_embeddings
        return emb_joined

    def get_tokenwise_mask(self, characterwise_mask: List[bool]) -> Dict[str, List[bool]]:
        tokenwise_masks = { }
        for k, t_embs, t_spans in ((k, self.tokenwise_embeddings[k], self.tokenwise_embedding_spans[k]) for k in self.tokenwise_embeddings):
            token_mask = [False] * len(t_embs)
            for i_t, (t_span_start, t_span_end) in enumerate(t_spans):
                if t_span_start != t_span_end: # Don't apply to SOS/EOS tokens
                    m = characterwise_mask[t_span_start:t_span_end]
                    assert all(m) or not any(m), 'Inconsistent mask'
                    if all(m):
                        token_mask[i_t] = True
            tokenwise_masks[k] = token_mask
        return tokenwise_masks


class EmbeddingDelta(nn.Module):
    def __init__(self, dims: Dict[str, int]) -> None:
        super().__init__()
        self.tokenwise_delta = nn.ParameterDict({ k: nn.Parameter(torch.zeros(d), requires_grad=True) for k, d in dims.items() })
    
    def apply(self, emb: PromptEmbedding, characterwise_mask: List[bool], alpha: float = 1.) -> PromptEmbedding:
        tokenwise_embeddings = { }
        matching_keys = [k for k in self.tokenwise_delta if k in emb.tokenwise_embeddings]
        assert len(matching_keys) > 0, f'Cannot apply delta if no embeddings match. Embeddings present in delta: {[k for k in self.tokenwise_delta]}. Embeddings present in embedding: {[k for k in emb.tokenwise_embeddings]}.'
        for k, t_embs, t_spans in ((k, emb.tokenwise_embeddings[k], emb.tokenwise_embedding_spans[k]) for k in matching_keys):
            token_mask = [0] * len(t_embs)
            for i_t, (t_span_start, t_span_end) in enumerate(t_spans):
                if t_span_start != t_span_end: # Don't apply to SOS/EOS tokens
                    m = characterwise_mask[t_span_start:t_span_end]
                    assert all(m) or not any(m), 'Inconsistent mask'
                    if all(m):
                        token_mask[i_t] = 1
            assert sum(token_mask) >= 1, f'No tokens in prompt selected for delta application via characterwise mask.'
            tokenwise_embeddings[k] = t_embs + alpha * torch.tensor(token_mask, dtype=t_embs.dtype, device=t_embs.device).unsqueeze(-1) * self.tokenwise_delta[k].unsqueeze(0).to(t_embs.dtype)
        return PromptEmbedding(
            prompt=emb.prompt,
            tokenwise_embeddings=tokenwise_embeddings,
            tokenwise_embedding_spans=emb.tokenwise_embedding_spans,
            pooled_embeddings=emb.pooled_embeddings
        )
