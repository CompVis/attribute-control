from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Tuple, Dict, Optional, List, Any
from pydoc import locate
import warnings

import torch
from torch import nn
from jaxtyping import Float, Integer
from PIL import Image
import diffusers
from diffusers import DDIMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from ..utils import reduce_tensors_recursively, broadcast_trailing_dims
from .. import PromptEmbedding, EmbeddingDelta


NAME_OPENAI_CLIP_VIT_L = 'oai_clip_vit_l'
NAME_OPENCLIP_G = 'openclip_g'


class ModelBase(ABC, nn.Module):
    def __init__(self, device: Union[str, torch.device] = 'cuda:0', compile: bool = False) -> None:
        super().__init__()

    @abstractproperty
    def dims(self) -> Dict[str, int]:
        raise NotImplementedError()

    @abstractmethod
    def embed_prompt(self, prompt: str) -> PromptEmbedding:
        raise NotImplementedError()
    
    @abstractmethod
    def sample(self, embs: List[PromptEmbedding], embs_neg: Optional[List[PromptEmbedding]], start_sample: Optional[Float[torch.Tensor, 'n c h w']] = None, start_after_relative: float = 0., cutoff_after_relative: float = 1., **kwargs) -> Union[List[Image.Image], Any]:
        raise NotImplementedError()

    def sample_delayed(self, embs: List[PromptEmbedding], embs_unmodified: List[PromptEmbedding], embs_neg: Optional[List[PromptEmbedding]], embs_neg_unmodified: Optional[List[PromptEmbedding]] = None, start_sample: Optional[Float[torch.Tensor, 'n c h w']] = None, delay_relative: float = .2, **kwargs) -> Union[List[Image.Image], Any]:
        if delay_relative == 0:
            return self.sample(embs=embs, embs_neg=embs_neg, start_sample=start_sample, **kwargs)
        else:
            raise NotImplementedError()

    def get_x_t(self, sample: Float[torch.Tensor, 'n c h w'], noise: Float[torch.Tensor, 'n c h w'], t_relative: float) -> Float[torch.Tensor, 'n c h w']:
        raise NotImplementedError()

    def predict_eps(self, embs: List[PromptEmbedding], start_sample: Float[torch.Tensor, 'n c h w'], t_relative: Float[torch.Tensor, 'n']) -> Float[torch.Tensor, 'n c h w']:
        raise NotImplementedError()


class DiffusersModelBase(ModelBase):
    def __init__(
        self,
        pipeline_type: str,
        model_name: str,
        pipe_kwargs: dict = { },
        device: Union[str, torch.device] = 'cuda:0',
        compile: bool = False,
    ) -> None:
        super().__init__(device=device, compile=compile)
        self.pipe: DiffusionPipeline = locate(pipeline_type).from_pretrained(model_name, **pipe_kwargs)
        self.pipe.to(device)
        assert isinstance(self.pipe, DiffusionPipeline)
        # self.pipe.safety_checker = None
        # self.pipe.set_progress_bar_config(disable=True)

        self._tokenizers = { }

    def _register_tokenizer(self, name: str, tokenizer, dim: int):
        self._tokenizers[name] = {
            'tokenizer': tokenizer,
            'vocab_reverse': { v: k for k, v in tokenizer.get_vocab().items() },
            'dim': dim,
        }

    @property
    def dims(self) -> Dict[str, int]:
        return { name: d['dim'] for name, d in self._tokenizers.items() }
    
    @torch.no_grad
    def get_token_spans(self, prompt: str) -> Dict[str, List[Tuple[int, int]]]:
        SPECIAL_TOKENS = ['<|startoftext|>', '<|endoftext|>']
        all_token_spans: Dict[str, List[Tuple[int, int]]] = { }
        for name, d_tokenizer in self._tokenizers.items():
            d = d_tokenizer['tokenizer'](prompt)
            token_ids = d['input_ids']
            tokens = [d_tokenizer['vocab_reverse'][i] for i in token_ids]

            # Rather primitive approach of figuring out which char in the unnormalized (!) prompt corresponds to which token
            token_spans = []
            i = 0
            skipped_chars = []
            for token in tokens:
                if token in SPECIAL_TOKENS:
                    token_spans.append((i, i))
                    continue
                if token.endswith('</w>'):
                    token = token[:-len('</w>')]
                while prompt[i] != token[0]:
                    skipped_chars.append(prompt[i])
                    i += 1
                assert i + len(token) <= len(prompt)
                assert prompt[i:(i + len(token))] == token, 'Misaligned'
                token_spans.append((i, i + len(token)))
                i += len(token)
            skipped_chars = ''.join(skipped_chars) + prompt[i:]
            assert not any(l.isalpha() for l in skipped_chars), f'Failed to assign some alphabetical characters to tokens for tokenizer {name}. Prompt: "{prompt}", unassigned characters: "{skipped_chars}".'

            all_token_spans[name] = token_spans

        return all_token_spans


class DiffusersSDModelBase(DiffusersModelBase):
    def __init__(
        self,
        pipeline_type: str,
        model_name: str,
        pipe_kwargs: dict = { },
        num_inference_steps: int = 50,
        device: Union[str, torch.device] = 'cuda:0',
        compile: bool = False,
    ) -> None:
        super().__init__(pipeline_type=pipeline_type, model_name=model_name, pipe_kwargs=pipe_kwargs, device=device, compile=compile)

        # Enable setting timesteps on the scheduler
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        set_timesteps_orig = self.pipe.scheduler.set_timesteps
        def set_timesteps_custom(num_inference_steps: int = None, timesteps: torch.Tensor = None, device: Union[str, torch.device] = None):
            if not timesteps is None:
                if isinstance(timesteps, torch.Tensor):
                    self.pipe.scheduler.timesteps = timesteps.to(device)
                else:
                    self.pipe.scheduler.timesteps = torch.from_numpy(timesteps).to(device)
            else:
                return set_timesteps_orig(num_inference_steps=num_inference_steps, device=device)
        self.pipe.scheduler.set_timesteps = set_timesteps_custom

        self.num_inference_steps = num_inference_steps

    @abstractmethod
    def _get_pipe_kwargs(self, embs: List[PromptEmbedding], start_sample: Optional[Float[torch.Tensor, 'n c h w']], **kwargs):
        raise NotImplementedError()

    @torch.no_grad
    def sample(self, embs: List[PromptEmbedding], embs_neg: Optional[List[PromptEmbedding]], start_sample: Optional[Float[torch.Tensor, 'n c h w']] = None, start_after_relative: float = 0., cutoff_after_relative: float = 1., **kwargs) -> Union[List[Image.Image], Any]:
        timesteps, _ = retrieve_timesteps(self.pipe.scheduler, kwargs.get('num_inference_steps', self.num_inference_steps), device=self.pipe.device, timesteps=kwargs.get('timesteps', None))
        timesteps = timesteps[int(round(start_after_relative * len(timesteps))):int(round(cutoff_after_relative * len(timesteps)))]
        return self.pipe(**(self._get_pipe_kwargs(embs, embs_neg, start_sample=start_sample, **kwargs) | { 'timesteps': timesteps, 'num_inference_steps': len(timesteps) })).images

    @torch.no_grad
    def sample_delayed(self, embs: List[PromptEmbedding], embs_unmodified: List[PromptEmbedding], embs_neg: Optional[List[PromptEmbedding]], embs_neg_unmodified: Optional[List[PromptEmbedding]] = None, start_sample: Optional[Float[torch.Tensor, 'n c h w']] = None, delay_relative: float = .2, **kwargs) -> Union[List[Image.Image], Any]:
        if delay_relative == 0:
            return self.sample(embs=embs, embs_neg=embs_neg, start_sample=start_sample, **kwargs)
        intermediate = self.sample(embs=embs_unmodified, embs_neg=(embs_neg_unmodified if not embs_neg_unmodified is None else embs_neg), start_sample=start_sample, **kwargs, output_type='latent', cutoff_after_relative=delay_relative)
        return self.sample(embs=embs, embs_neg=embs_neg, start_sample=intermediate, **kwargs, start_after_relative=delay_relative)

    def _get_eps_pred(self, t: Integer[torch.Tensor, 'n'], sample: Float[torch.Tensor, 'n ...'], model_output: Float[torch.Tensor, 'n ...']) -> Float[torch.Tensor, 'n ...']:
        alpha_prod_t = broadcast_trailing_dims(self.pipe.scheduler.alphas_cumprod[t.to(self.pipe.scheduler.alphas_cumprod.device)].to(model_output.device), model_output)
        beta_prod_t = 1 - alpha_prod_t
        if self.pipe.scheduler.config.prediction_type == "epsilon":
            return (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.pipe.scheduler.config.prediction_type == "sample":
            return model_output
        elif self.pipe.scheduler.config.prediction_type == "v_prediction":
            return (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise NotImplementedError(f'Missing implementation for {self.pipe.scheduler.config.prediction_type=}.')

    def get_x_t(self, sample: Float[torch.Tensor, 'n c h w'], noise: Float[torch.Tensor, 'n c h w'], t_relative: Float[torch.Tensor, 'n']) -> Float[torch.Tensor, 'n c h w']:
        t = torch.round(t_relative * (self.num_inference_steps - 1)).to(torch.int64)
        return self.pipe.scheduler.add_noise(sample, noise, t.unsqueeze(-1))


class SD15(DiffusersSDModelBase):
    def __init__(
        self,
        pipeline_type: str,
        model_name: str,
        num_inference_steps: int = 50,
        pipe_kwargs: dict = { },
        device: Union[str, torch.device] = 'cuda:0',
        compile: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(pipeline_type=pipeline_type, model_name=model_name, num_inference_steps=num_inference_steps, pipe_kwargs=pipe_kwargs, device=device, compile=compile)

        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        d_v_major, d_v_minor, *_ = diffusers.__version__.split('.')
        if int(d_v_major) > 0 or int(d_v_minor) >= 25:
            self.pipe.fuse_qkv_projections()
        if gradient_checkpointing:
            if compile:
                warnings.warn('Gradient checkpointing is typically not compatible with compiling the U-Net. This will likely lead to a crash.')
            self.pipe.unet.enable_gradient_checkpointing()
        if compile:
            assert int(d_v_major) > 0 or int(d_v_minor) >= 25, 'Use at least diffusers==0.25 to enable proper functionality of torch.compile().'
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.vae.to(memory_format=torch.channels_last)
            self.pipe.unet = torch.compile(self.pipe.unet, mode=None, fullgraph=True)
        self.pipe.unet.train()
        self.pipe.unet.requires_grad_(False)

        self._register_tokenizer(NAME_OPENAI_CLIP_VIT_L, self.pipe.tokenizer, 768)

    @torch.no_grad
    def embed_prompt(self, prompt: str) -> PromptEmbedding:
        token_spans = self.get_token_spans(prompt)
        prompt_embeds, _ = self.pipe.encode_prompt(prompt=[prompt], device=self.pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        return PromptEmbedding(
            prompt=prompt,
            tokenwise_embeddings={ NAME_OPENAI_CLIP_VIT_L: prompt_embeds[0] },
            tokenwise_embedding_spans=token_spans,
        )

    def _get_pipe_kwargs(self, embs: List[PromptEmbedding], embs_neg: Optional[List[PromptEmbedding]] = None, start_sample: Optional[torch.Tensor] = None, **kwargs):
        return {
            'prompt_embeds': reduce_tensors_recursively(*[emb.tokenwise_embeddings[NAME_OPENAI_CLIP_VIT_L] for emb in embs], reduction_op=torch.stack),
            'negative_prompt_embeds': reduce_tensors_recursively(*[emb.tokenwise_embeddings[NAME_OPENAI_CLIP_VIT_L] for emb in embs_neg], reduction_op=torch.stack) if (not embs_neg is None) and not any(e is None for e in embs_neg) else None,
        } | ({ 'latents': start_sample } if not start_sample is None else { }) | kwargs

    def predict_eps(self, embs: List[PromptEmbedding], start_sample: Float[torch.Tensor, 'n c h w'], t_relative: Float[torch.Tensor, 'n']) -> Float[torch.Tensor, 'n c h w']:
        i_t = torch.round(t_relative * (self.num_inference_steps - 1)).to(torch.int64)
        t = self.pipe.scheduler.timesteps[i_t]
        return self._get_eps_pred(t, start_sample, self.pipe.unet(start_sample, t, encoder_hidden_states=self._get_pipe_kwargs(embs, embs_neg=None, start_sample=None)['prompt_embeds']).sample)


class SDXL(SD15):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.pipe.text_encoder_2.requires_grad_(False)
        self._register_tokenizer(NAME_OPENCLIP_G, self.pipe.tokenizer_2, 1280)

    @torch.no_grad
    def embed_prompt(self, prompt: str) -> PromptEmbedding:
        token_spans = self.get_token_spans(prompt)
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(prompt=[prompt], device=self.pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        return PromptEmbedding(
            prompt=prompt,
            tokenwise_embeddings={ NAME_OPENAI_CLIP_VIT_L: prompt_embeds[0][...,:768], NAME_OPENCLIP_G: prompt_embeds[0][...,768:] },
            tokenwise_embedding_spans=token_spans,
            pooled_embeddings={ NAME_OPENAI_CLIP_VIT_L: pooled_prompt_embeds[0][:768], NAME_OPENCLIP_G: pooled_prompt_embeds[0][768:] },
        )

    def _get_pipe_kwargs(self, embs: List[PromptEmbedding], embs_neg: Optional[List[PromptEmbedding]] = None, start_sample: Optional[torch.Tensor] = None, **kwargs):
        return {
            'prompt_embeds': torch.cat([
                reduce_tensors_recursively(*[emb.tokenwise_embeddings[NAME_OPENAI_CLIP_VIT_L] for emb in embs], reduction_op=torch.stack),
                reduce_tensors_recursively(*[emb.tokenwise_embeddings[NAME_OPENCLIP_G] for emb in embs], reduction_op=torch.stack)
            ], dim=-1),
            'pooled_prompt_embeds': torch.cat([
                reduce_tensors_recursively(*[emb.pooled_embeddings[NAME_OPENAI_CLIP_VIT_L] for emb in embs], reduction_op=torch.stack),
                reduce_tensors_recursively(*[emb.pooled_embeddings[NAME_OPENCLIP_G] for emb in embs], reduction_op=torch.stack)
            ], dim=-1),
            'negative_prompt_embeds': torch.cat([
                reduce_tensors_recursively(*[emb.tokenwise_embeddings[NAME_OPENAI_CLIP_VIT_L] for emb in embs_neg], reduction_op=torch.stack),
                reduce_tensors_recursively(*[emb.tokenwise_embeddings[NAME_OPENCLIP_G] for emb in embs_neg], reduction_op=torch.stack)
            ], dim=-1) if (not embs_neg is None) and not any(e is None for e in embs_neg) else None,
            'negative_pooled_prompt_embeds': torch.cat([
                reduce_tensors_recursively(*[emb.pooled_embeddings[NAME_OPENAI_CLIP_VIT_L] for emb in embs_neg], reduction_op=torch.stack),
                reduce_tensors_recursively(*[emb.pooled_embeddings[NAME_OPENCLIP_G] for emb in embs_neg], reduction_op=torch.stack)
            ], dim=-1) if (not embs_neg is None) and not any(e is None for e in embs_neg) else None,
        } | ({ 'latents': start_sample } if not start_sample is None else { }) | kwargs

    def _compute_time_ids(self, device, weight_dtype) -> torch.Tensor:
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (1024, 1024)
        target_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(device, dtype=weight_dtype)
        return add_time_ids

    def predict_eps(self, embs: List[PromptEmbedding], start_sample: Float[torch.Tensor, 'n c h w'], t_relative: Float[torch.Tensor, 'n']) -> Float[torch.Tensor, 'n c h w']:
        i_t = torch.round(t_relative * (self.num_inference_steps - 1)).to(torch.int64)
        self.pipe.scheduler.set_timesteps(self.num_inference_steps)
        t = self.pipe.scheduler.timesteps[i_t.to(self.pipe.scheduler.timesteps.device)].to(start_sample.device)

        p_embs = self._get_pipe_kwargs(embs, embs_neg=None, start_sample=None)
        add_time_ids = self._compute_time_ids(start_sample.device, start_sample.dtype)
        add_time_ids = add_time_ids.to(start_sample.device).repeat(len(embs), 1)
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": p_embs['pooled_prompt_embeds'],
        }
        return self._get_eps_pred(t, start_sample, self.pipe.unet(start_sample, t, encoder_hidden_states=p_embs['prompt_embeds'], added_cond_kwargs=unet_added_conditions).sample)


class StableCascade(DiffusersModelBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.pipe.prior_pipe.to(self.pipe.device)
        self.pipe.decoder_pipe.to(self.pipe.device)

        self._tokenizers = { }
        self._register_tokenizer(NAME_OPENCLIP_G, self.pipe.tokenizer, 1280)

    @torch.no_grad
    def embed_prompt(self, prompt: str) -> PromptEmbedding:
        token_spans = self.get_token_spans(prompt)
        prompt_embeds, pooled_prompt_embeds, _, _ = self.pipe.prior_pipe.encode_prompt(prompt=[prompt], device=self.pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False, batch_size=1)
        return PromptEmbedding(
            prompt=prompt,
            tokenwise_embeddings={ NAME_OPENCLIP_G: prompt_embeds[0] },
            tokenwise_embedding_spans=token_spans,
            pooled_embeddings={ NAME_OPENCLIP_G: pooled_prompt_embeds[0] },
        )

    def _get_pipe_kwargs(self, embs: List[PromptEmbedding], embs_neg: Optional[List[PromptEmbedding]] = None, start_sample: Optional[torch.Tensor] = None, **kwargs):
        return {
            'prompt_embeds': reduce_tensors_recursively(*[emb.tokenwise_embeddings[NAME_OPENCLIP_G] for emb in embs], reduction_op=torch.stack),
            'prompt_embeds_pooled': reduce_tensors_recursively(*[emb.pooled_embeddings[NAME_OPENCLIP_G] for emb in embs], reduction_op=torch.stack),
            'negative_prompt_embeds': reduce_tensors_recursively(*[emb.tokenwise_embeddings[NAME_OPENCLIP_G] for emb in embs_neg], reduction_op=torch.stack) if (not embs_neg is None) and not any(e is None for e in embs_neg) else None,
            'negative_prompt_embeds_pooled': reduce_tensors_recursively(*[emb.pooled_embeddings[NAME_OPENCLIP_G] for emb in embs_neg], reduction_op=torch.stack)
             if (not embs_neg is None) and not any(e is None for e in embs_neg) else None,
        } | ({ 'latents': start_sample } if not start_sample is None else { }) | kwargs

    @torch.no_grad
    def sample(self, embs: List[PromptEmbedding], embs_neg: Optional[List[PromptEmbedding]], start_sample: Optional[Float[torch.Tensor, 'n c h w']] = None, start_after_relative: float = 0., cutoff_after_relative: float = 1., **kwargs) -> Union[List[Image.Image], Any]:
        assert start_after_relative == 0 and cutoff_after_relative == 1, 'Not implemented (yet).'
        return self.pipe(width=1024, height=1024, **self._get_pipe_kwargs(embs, embs_neg, start_sample=start_sample, **kwargs)).images
