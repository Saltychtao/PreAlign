from transformer.trainer_utils import seed_worker
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from typing import Dict
import random
import os
import time

from src.trainer.trainer import (
    Trainer,
    Optional
)

from src.trainer.lm_trainer import Trainer
from src.trainer.trainer_utils import parse_layers, build_metric, ContrastiveMetric, AllGatherGrad

class BilingualDictionaryPretrainer(Trainer):
    def __init__(self,args,**kwargs):
        super().__init__(args=args,**kwargs)
        self.metric= ContrastiveMetric(tau=0.1)
        self.alpha = args.repre_alignment_strengthsda

    def _save(self, output_dir):
        super()._save(output_dir, state_dict)

    def compute_pairwise_alignment(self, x_states, y_states):
        x,y = x_states.float(), y_states.float()
        L,B,H = x.size()
        x = x.reshape(L*B, H)
        y = y.reshape(L*B,H)
        cosine = F.consine_similarity(x,y)
        return cosine.reshape(L,B).mean(dim=1)

    def compute_loss(self,model,inputs,return_outputs=False):
        loss = 0
        output = model(**inputs, output_hidden_states=True)
        x_mask, y_mask = inputs["attention_mask"].chunk(2,dim=0)

        x_hidden_states, y_hidden_states = [],[]
        for layer_hidden in output.hidden_states:
            _x, _y = layer_hidden.chunk(2,dim=0)
            _x = _x.masked_fill(~x_mask.unsqueeze(-1),0).sum(dim=1)
            _y = _y.masked_fill(~x_mask.unsqueeze(-1),0).sum(dim=1)
            x_hidden_states.append(_x)
            y_hidden_states.append(_y)

        x_hidden_states = torch.stack(x_hidden_states,dim=0)
        y_hidden_states = torch.stack(y_hidden_states,dim=0)

        x_input_ids,y_input_ids = inputs["input_ids"].chunk(2,dim=0)
        x_ouptut_embed = model.output_embed(x_input_ids).masked_fill(~x_mask.unsqueeze(-1),0).sum(dim=1)
        y_ouptut_embed = model.output_embed(y_input_ids).masked_fill(~y_mask.unsqueeze(-1),0).sum(dim=1)

        x_states = torch.cat((x_hidden_states, x_ouptut_embed.unsqueeze(0)),dim=0)
        y_states = torch.cat((y_hidden_states, y_ouptut_embed.unsqueeze(0)),dim=0)

        x_states = AllGatherGrad.apply(x_states)
        y_states = AllGatherGrad.apply(y_states)

        N,L,B,H = x_states.size()
        x_states = x_states.transpose(0,1).reshape(L,N*B,H)
        y_states = y_states.transpose(0,1).reshape(L,N*B,H)

        alignment_loss = self.metric(x_states, y_states)

        loss = self.alpha * alignment_loss + output.loss

        if not model.training:
            return ({"loss": loss}, output) if return_outputs else {"loss": loss}
        else:
            return {
                "loss": loss,
                "lm_loss": output.loss,
                "alignment_loss": alignment_loss
            }


