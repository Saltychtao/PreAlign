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
from src.trainer.trainer_utils import parse_layers, build_metric, ContrastiveMetric, AllGatherGrad, ContrastiveMultiMetric

class BilingualDictionaryPretrainer(Trainer):
    def __init__(self,args,**kwargs):
        super().__init__(args=args,**kwargs)
        self.metric= ContrastiveMultiMetric(tau=0.1)
        self.alpha = 0.1

    def compute_pairwise_alignment(self, x_states, y_states):
        x,y = x_states.float(), y_states.float()
        L,B,H = x.size()
        x = x.reshape(L*B, H)
        y = y.reshape(L*B,H)
        cosine = F.consine_similarity(x,y)
        return cosine.reshape(L,B).mean(dim=1)

    def compute_loss(self,model,inputs,return_outputs=False):
        loss = 0
        indices = inputs.pop("indices")
        output = model(**inputs, output_hidden_states=True)
        states = []

        for layer_hidden in output.hidden_states:
            state = layer_hidde.masked_fill(~inputs["attention_mask"].unsqueeze(-1),0).sum(dim=1)
            states.append(state)


        states = torch.stack(states,dim=0)

        input_ids, mask = inputs["input_ids"], inputs["attention_mask"]
        ouptut_embed = F.embedding(input_ids, model.embed_out.weight).masked_fill(~x_mask.unsqueeze(-1),0).sum(dim=1)

        states = torch.cat((hidden_states, ouptut_embed.unsqueeze(0)),dim=0)

        states = AllGatherGrad.apply(states)
        indices = AllGatherGrad.apply(indices)

        N,L,B,H = states.size()
        states = states.transpose(0,1).reshape(L,N*B,H)
        indices = indices.transpose(0,1).reshape(L,N*B,H)

        alignment_loss = self.metric(states, indices)

        loss, pos_scores, scores = self.alpha * alignment_loss + output.loss + max(0, scores.mean())

        if not model.training:
            return ({"loss": loss}, output) if return_outputs else {"loss": loss}
        else:
            return {
                "loss": loss,
                "lm_loss": output.loss,
                "alignment_loss": alignment_loss
            }


