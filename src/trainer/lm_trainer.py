from src.trainer.trainer import Trainer

class LMTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        if not model.training:
            return ({"loss": loss},outputs) if return_outputs else {"loss":loss}
        else:
            return {
                "loss": loss,
                "lm_loss": outputs.loss
            }