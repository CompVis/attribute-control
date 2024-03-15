from typing import List, Dict
from itertools import product
from pathlib import Path
from tqdm.auto import tqdm
import hydra
import torch
from omegaconf import DictConfig

from attribute_control import EmbeddingDelta
from attribute_control.model import ModelBase
from attribute_control.prompt_utils import get_mask_regex


@hydra.main(config_path="configs", config_name="learn_delta_naive_clip")
@torch.no_grad()
def main(cfg: DictConfig):
    print(cfg)
    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model
    prompts: List[Dict[str, str]] = cfg.prompts
    prefixes: List[str] = cfg.prefixes

    # Compute the deltas for each prompt pair
    deltas = []
    for prefix, d_prompt in tqdm(product(prefixes, prompts), total=(len(prefixes) * len(prompts))):
        target_token_embs = { }
        for direction in ['prompt_positive', 'prompt_negative']:
            emb = model.embed_prompt(f'{prefix} {d_prompt[direction]}')
            tokenwise_masks = emb.get_tokenwise_mask(get_mask_regex(emb.prompt, d_prompt[direction]))
            # Retrieve last token that is part of the target word
            target_token_embs[direction] = { encoder: embedding[len(tokenwise_masks[encoder]) - 1 - tokenwise_masks[encoder][::-1].index(True)] for encoder, embedding in emb.tokenwise_embeddings.items() }
        # Eq. 2
        deltas.append({ encoder: target_token_embs['prompt_positive'][encoder] - target_token_embs['prompt_negative'][encoder] for encoder in emb.tokenwise_embeddings })

    # Compute the average delta
    delta = EmbeddingDelta(model.dims)
    for encoder in delta.tokenwise_delta:
        delta.tokenwise_delta[encoder].copy_(torch.stack([d[encoder] for d in deltas]).mean(dim=0))

    output_dir = Path('./checkpoints')
    checkpoint_path = output_dir / f'delta.pt'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'delta': delta.cpu().state_dict(),
    }, checkpoint_path)


if __name__ == "__main__":
    main()
