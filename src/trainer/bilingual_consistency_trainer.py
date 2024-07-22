import torch.nn.functional as F
import torch

from src.trainer.trainer import Trainer
from src.trainer.trainer_utils import ContrastiveMetric

class BilingualConsistencyTrainer(Trainer):
    def __init__(self,args,**kwargs):
        super().__init__(args=args,**kwargs)

        if len(args.consistency_patterns) != len(args.consistency_metrics):
            raise ValueError("The number of consistency patterns should be equal to the consistency metrics.")
        self.consistency_strength = args.consistency_strength
        self.constrastive_metric = ContrastiveMetric(tau=0.07,p=2)


    def compute_loss(self, model, inputs, return_outputs=False):
        lm_loss = super().compute_loss(model,inputs,return_outputs=False)

        source_outputs = model(**inputs["source"],output_hidden_states=True)
        target_outputs = model(**inputs["target"],output_hidden_states=True)
        src_lengths = inputs["source"]["attention_mask"].mean(dim=-1,keepdim=True)
        tgt_lengths = inputs["target"]["attention_mask"].mean(dim=-1,keepdim=True)

        src_repres = []
        tgt_repres = []
        for src_layer_repre, tgt_layer_repre in zip(source_outputs.hidden_states,target_outputs.hidden_states):
            # B,T,H
            src_repres.append(src_layer_repre.div(src_lengths))
            tgt_repres.append(tgt_layer_repre.div(tgt_lengths))

        src_repres = torch.stack(src_repres,dim=0)
        tgt_repres = torch.stack(tgt_repres,dim=0) # L,B,H
        contrastive_loss = self.constrastive_metric(src_repres,tgt_repres)

        return lm_loss + self.consistency_strength * contrastive_loss





        



