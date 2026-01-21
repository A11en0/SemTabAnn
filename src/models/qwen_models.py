"""
Qwen3 Adapter for replacing BERT in the Doduo framework
"""
import torch
import torch.nn as nn
from typing import Optional
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPast, SequenceClassifierOutput
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel, Qwen3Model, Qwen3Config
from transformers.cache_utils import Cache

import logging
logger = logging.getLogger(__name__)


class Qwen3ForMultiOutputClassification(Qwen3PreTrainedModel):    
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.model = Qwen3Model(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        
        # Add learnable prototypes for PiCO loss
        self.post_init()
        self.prototypes = nn.Parameter(torch.randn(self.num_labels, config.hidden_size))
        nn.init.xavier_uniform_(self.prototypes)
        
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_features: Optional[bool] = False,
        return_dict: Optional[bool] = None
    ) -> SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.classifier(hidden_states)

        loss = None

        if return_features:
            return (logits, hidden_states, self.prototypes)
        else:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )

    # def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # return {"input_ids": input_ids, **kwargs}
        