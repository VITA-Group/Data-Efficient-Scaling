import math

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
from torch import nn
from transformers import (
    AutoConfig,
    BertConfig
)

from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaLMHead
from bert_model import BertForMaskedLM, BertOnlyMLMHead

BertLayerNorm = torch.nn.LayerNorm


# The GLUE function is copied from huggingface transformers:
# https://github.com/huggingface/transformers/blob/c6acd246ec90857b70f449dcbcb1543f150821fc/src/transformers/activations.py
def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


class BertSharedHead(BertOnlyMLMHead):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__(config)
        self.do_voken_cls = config.do_voken_cls
        self.do_voken_ctr = config.do_voken_ctr

        assert int(self.do_voken_cls) + int(self.do_voken_ctr) == 1
        if self.do_voken_cls:
            self.visn_decoder = nn.Linear(config.hidden_size, config.voken_size, bias=True)

        if self.do_voken_ctr:
            self.visn_decoder = nn.Linear(config.voken_dim, config.hidden_size, bias=True)

    def forward(self, features, **kwargs):
        """
        :param features: [batch, length, dim]
        :return: lang_scores [batch, length, vocab_size],
                 visn_scores [batch, length, voken_size]
        """
        x = self.predictions.transform(features)    # batch_size, length, dim

        lang_scores = self.predictions.decoder(x) + self.predictions.bias

        if self.do_voken_cls:
            visn_scores = self.visn_decoder(x)
        elif self.do_voken_ctr:
            voken_feats = kwargs['voken_feats']
            y = self.visn_decoder(voken_feats)  # voken_size, dim
            visn_scores = torch.einsum('bik,jk->bij', x, y)
        else:
            assert False

        return lang_scores, visn_scores

class SimpleBertForMaskedLM(BertForMaskedLM):

    def __init__(self, config, args=None):
        super().__init__(config, args=args)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output)
        loss_fct = CrossEntropyLoss()
        token_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        return {'loss': token_loss, 'lm_loss': token_loss}


class SimpleRobertaForMaskedLM(RobertaForMaskedLM):

    def __init__(self, config, args=None):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            **kwargs
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0]

        prediction_scores = self.lm_head(sequence_output)
        loss_fct = CrossEntropyLoss()
        token_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        return {'loss': token_loss, 'lm_loss': token_loss}
