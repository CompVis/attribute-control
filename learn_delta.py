from typing import List, Dict
from itertools import product
from pathlib import Path
from pydoc import locate
import logging

from tqdm.auto import tqdm
import hydra
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import einops
from omegaconf import DictConfig

from attribute_control import EmbeddingDelta
from attribute_control.model import ModelBase
from attribute_control.model.model import DiffusersModelBase
from attribute_control.prompt_utils import get_mask_regex


class PromptCombinationDataset(Dataset):
    def __init__(self, prompts: Dict[str, str], prefixes: List[str]) -> None:
        super().__init__()
        self.prompts = prompts
        self.prefixes = prefixes
    
    def __len__(self) -> int:
        return len(self.prompts) * len(self.prefixes)
    
    def __getitem__(self, index) -> Dict[str, str]:
        i_prompt = index // len(self.prefixes)
        prefix = self.prefixes[index - i_prompt * len(self.prefixes)]
        return { k: (f'{prefix} {v}' if 'prompt' in k else v) for k, v in self.prompts[i_prompt].items() }


@hydra.main(config_path="configs", config_name="learn_delta")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    print(cfg)
    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model
    if isinstance(model, DiffusersModelBase):
        model.pipe.set_progress_bar_config(disable=True)
    dataset = PromptCombinationDataset(cfg.prompts, cfg.prefixes)

    batch_size: int = cfg.batch_size
    scale_batch_size: int = cfg.scale_batch_size
    scale_min, scale_max = cfg.scale_range
    randomize_scale_sign: bool = cfg.randomize_scale_sign
    grad_accum_steps: int = cfg.grad_accum_steps
    max_steps: int = cfg.max_steps
    ckpt_logging_freq: int = cfg.ckpt_logging_freq

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    delta = EmbeddingDelta(model.dims)
    if not cfg.init_from_ckpt is None:
        logger.info(f'Loading checkpoint from {cfg.init_from_ckpt}.')
        state_dict = torch.load(cfg.init_from_ckpt)
        delta.load_state_dict(state_dict['delta'])
    delta = delta.to(cfg.device)

    optimizer = locate(cfg.optim_class)(delta.parameters(), **cfg.optim_params)
    ckpt_output_dir = Path('./checkpoints')

    data_iter = iter(dataloader)
    logger.info(f'Starting optimization...')
    for global_step in (pbar := tqdm(range(max_steps), desc='Optimizing')):
        loss_sum = 0
        for accum_step in range(grad_accum_steps):
            with torch.no_grad():
                if (batch := next(data_iter, None)) is None:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                prompts_embedded = { k: [model.embed_prompt(v) for v in vs] for k, vs in batch.items() if 'prompt' in k }
                t_relative = torch.rand((batch_size,), device=cfg.device)
                if batch_size != 1:
                    x_0 = model.sample(prompts_embedded['prompt_target'], embs_neg=None, guidance_scale=cfg.base_sample_settings.guidance_scale, output_type='latent')
                    x_t = model.get_x_t(x_0, torch.randn_like(x_0), t_relative)
                else:
                    x_t = model.sample(prompts_embedded['prompt_target'], embs_neg=None, cutoff_after_relative=float(t_relative.item()), guidance_scale=cfg.base_sample_settings.guidance_scale, output_type='latent')
                eps_p = model.predict_eps(prompts_embedded['prompt_positive'], x_t, t_relative).unsqueeze(0)
                eps_t = model.predict_eps(prompts_embedded['prompt_target'], x_t, t_relative).unsqueeze(0)
                eps_n = model.predict_eps(prompts_embedded['prompt_negative'], x_t, t_relative).unsqueeze(0)
                x_t = x_t.unsqueeze(0).expand(scale_batch_size, -1, -1, -1, -1)
                scale = (((torch.rand((scale_batch_size, batch_size), device=x_t.device) > .5).float() * 2 - 1) if randomize_scale_sign else False) * ((scale_max - scale_min) * torch.rand((scale_batch_size, batch_size), device=x_t.device) + scale_min)
                scale_cpu = scale.cpu()
                eps_target = eps_t + scale.view(scale_batch_size, batch_size, 1, 1, 1) * (eps_p - eps_n)

            # TODO: check that flattening order is the same for the embeddings and x_t
            eps_delta = model.predict_eps(
                [delta.apply(emb, get_mask_regex(emb.prompt, batch['pattern_target'][i_p]), scale_cpu[i_s, i_p]) for i_p, emb in enumerate(prompts_embedded['prompt_target']) for i_s in range(scale_batch_size)],
                einops.rearrange(x_t, 'b_e b c h w -> (b_e b) c h w').detach(),
                t_relative.unsqueeze(0).expand(scale_batch_size, -1).flatten()
            ).view(*x_t.shape)
            loss = F.mse_loss(eps_delta, eps_target.detach())

            loss.backward()
            loss_sum += float(loss.detach().item())
            pbar.set_postfix({ 'loss': loss_sum / (accum_step + 1) })

        # pbar.set_description(f'Optimizing. Loss: {loss_sum / grad_accum_steps:.6f}')
        pbar.set_postfix({ 'loss': loss_sum / grad_accum_steps })
        optimizer.step()
        optimizer.zero_grad()

        if (global_step + 1) % ckpt_logging_freq == 0:
            checkpoint_path = ckpt_output_dir / f'delta_step_{global_step + 1}.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'delta': delta.state_dict(),
            }, checkpoint_path)
            logger.info(f'Saved intermediate checkpoint to {checkpoint_path}.')

    checkpoint_path = ckpt_output_dir / f'delta.pt'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'delta': delta.cpu().state_dict(),
    }, checkpoint_path)
    logger.info(f'Saved final checkpoint to {checkpoint_path}.')


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
