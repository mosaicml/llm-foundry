# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Mapping

import torch
from composer.utils import dist
from transformers import AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmfoundry.models.llm_embed.modeling_llm_embed import ContrastiveModel


class FinetuneEmbeddingModel(ContrastiveModel):

    def construct_model(self) -> CausalLMOutputWithPast:
        # Define the model construction specific to FinetuneEmbeddingModel
        model = None

        def load_model():
            return AutoModel.from_pretrained(
                self.pretrained_model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token,
                **self.kwargs,
            )

        if dist.get_global_rank() == 0:
            model = load_model()
        dist.barrier()
        if model is None:
            model = load_model()

        assert model, 'Model is not loaded properly'
        return model

    def get_hidden_state(self, outputs: CausalLMOutputWithPast) -> torch.Tensor:
        """Override to return the last hidden state."""
        return outputs.last_hidden_state

    def handle_language_head(
        self,
        outputs: CausalLMOutputWithPast,
    ) -> torch.Tensor:
        """Override to skip language head handling."""
        return torch.tensor(
            0,
            dtype=torch.float32,
            device=outputs.last_hidden_state.device,
        )

    def flops_per_batch(self, batch: Mapping) -> int:
        # Get the batch size and maximum sequence length
        bs, msl = batch['input_ids'].shape[0:2]

        model_dimension = self._get_attribute(
            self.model.config,
            [
                'hidden_size',
                'd_model',
                'n_embd',
                'dim',
                'embed_dim',
                'embedding_size',
                'hidden_dim',
            ],
        )

        num_layers = self._get_attribute(
            self.model.config,
            [
                'num_hidden_layers',
                'n_layer',
                'num_layers',
                'encoder_layers',
                'decoder_layers',
                'n_layers',
                'num_blocks',
                'layer_count',
            ],
        )

        num_parameters = sum(p.numel() for p in self.model.parameters())

        # Estimate FLOPs
        params_flops = 2 * num_parameters
        seq_flops = params_flops * msl
        attn_flops = bs * num_layers * 2 * msl * model_dimension
        total_flops = seq_flops * bs + attn_flops

        return total_flops

    def _get_attribute(self, config: Any, possible_names: list):
        """Retrieve an attribute from config using a list of possible names."""
        for name in possible_names:
            value = getattr(config, name, None)
            if value is not None:
                return value
        return None
