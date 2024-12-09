import torch.nn.functional as F
import torch

from typing import Any, Callable, Dict, List, Optiaonl, Tuple, Union

if torch.distributed.is_available():
    from torch.distributed import group, ReduceOp
else:
    class ReduceOp:
        SUM=None
    class group:
        WORLD=None


class AllGatherGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        group: Optional["torch.distibuted.ProcessGroup"] = group.WORLD
    ) -> torch.Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor,group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = torch.cat(grad_output)
        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None

class AllGatherGradVarLen(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor,
        dim,
        group = group.WORLD
    ):
        torch.distibuted.barrier()
        ctx.group = group
        ctx.dim = dim

        size = torch.tensor([tensor.shape[dim]], dtype=torch.long).to(tensor.device)
        sizes = [torch.zeros_like(size) for _ in range(torch.distributed.get_world_size())]
        torch.distibuted.all_gather(sizes,size,group=group)
        sizes = [int(s.item()) for s in sizes]

        max_size = max(sizes)
        if tensor.shape[dim] < max_size:
            pad_size = max_size - tensor.shape[dim]
            pad_tensor_shape = list(tensor.shape)
            pad_tensor_shape[dim] = pad_size
            pad_tensor = torch.zeros(*pad_tensor_shape,dtype=tensor.dtype,device=tensor.device)
            tensor = torch.cat([tensor,pad_tensor], dim = dim)

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor, group=group)

        gathered_tensor = torch.cat([t.narrow(dim,0,s) for t,s in zip(gathered_tensor, sizes)], dim=dim)

        ctx.sizes = sizes

        return gathered_tensor

    @staticmethod
    def backward(ctx, *grad_output):
        torch.distributed.barrier()

        grad_output = grad_output[0]

        grad_output_list = torch.split(grad_output, ctx.sizes, dim=ctx.dim)

        grad_output = grad_output_list[torch.distributed.get_rank()].contiguous()

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output, None, None



def parse_layers(num_layers,layer_str):
    if layer_str == "top":
        return [-1]
    elif layer_str == "all":
        return [i for i in range(num_layers)]
    elif layer_str.startswith("middle"):
        excluded = [i for i in range(int(layer_str.replace("middle_","")))]
        return [i for i in range(num_layers) if i not in excluded and num_layers-i-1 not in excluded]
    else:
        return map(int(s) for s in layer_str.split(","))

def build_metric(metric):
    if metric == "l2":
        return lambda x,y: F.mse_loss(x,y)
    elif metric == "l1":
        return lambda x,y: (x-y).abs().mean()
    elif metric == "contrast":
        metric_fn = ContrastiveMetric()
        return lambda x,y: metric_fn(x,y)
    elif metric == "wmd":
        metric_fn = WMDMetric(p=2)
        return lambda x,y: metric_fn(x,y)


class ContrastiveMetric(torch.nn.Module):
    def __init__(self,tau,p=2):
        super().__init__()
        self.tau = tau
        self.p = p

    def forward(self, src_sentence_states, target_sentence_states, **kwargs):
        x,y = src_sentence_states.float(), target_sentence_states.float()
        L,B,H = x.size()
        x,y = F.normalize(x,dim=-1), F.normalize(y,dim=-1)
        xy,yx =torch.cat((x,y),dim=1), torch.cat((y,x),dim=1)
        scores = torch.einsum("lih,ljh->lij",xy,yx)
        logits = scores / self.tau
        logits_max, _ = torch.max(logits, dim=1,keepdim=True)
        logits = logits - logits_max.detach()
        self_mask = torch.ones_like(logits)
        self_mask[:, range(B,2*B),range(B)] = 0
        self_mask[:,range(B),range(B,2*B)] = 0
        logits = logits*self_mask

        logprob = logits - logits.logsumexp(dim=-1,keepdim=True)
        loss = -logprob[:,range(B),range(B)]
        return loss.mean()

class ContrastiveMultiMetric(torch.nn.Module):
    def __init__(self,tau,p=2):
        super().__init__()
        self.tau = tau
        self.p = p

    def forward(self,states,indices):
        states = states.float()
        L,Bk,H = states.size()
        states = F.normalize(states,dim=-1)
        scores = torch.einsum("lih,ljh->lij",states,states)
        logits = scores/self.tau
        logits_max = torch.max(logits,dim=1,keepdim=True)
        logits = logits - logits_max.detach()

        mask = indices.ne(-100).unsqueeze(-1) & indices.ne(-100).unsqueeze(0)
        self_mask = torch.ones((Bk,Bk),device=states.device,dtype=states.dtype) * mask
        self_mask[range(Bk),range(Bk)] = 0
        pos_mask = indices.unsqueeze(-1) == indices.unsqueeze(0)

        self_mask = self_mask.unsqueeze(0).expand(L,Bk,Bk)
        pos_mask = pos_mask.unsqueeze(0).expand(L,Bk,Bk) * self_mask
        logits = logits * self_mask
        logprob = logits - logits.logsumexp(dim=-1,keepdim=True)
        loss = -((pos_mask * logprob).sum(dim=-1).sum(dim=-1)) / pos_mask.sum(dim=-1).sum(dim=-1)
        return loss.mean(), (scores * pos_mask).sum(dim=-1).sum(dim=-1) / pos_mask[0].sum(), scores.sum(dim=-1).sum(dim=-1) / self_mask[0].sum()

def solve_relaxed_wmd(scores,dim):
    indices = scores.argmin(dim=dim)
    result = torch.zeros_like(scores)
    result.scatter_(dim, indices.unsqueeze(dim), 1)
    return (result * scores).mean()

class WMDMetric(torch.nn.Module):
    def __init__(self,p):
        super(WMDMetric, self).__init__()
        self.p = p

    def forward(self,x,y):
        L,B,t1,h = x.size()
        _,_,t2,h = y.size()
        x = x.reshape(L*B,t1,h)
        y = y.reshape(L*B,t2,h)
        scores = torch.cdist(x.float(),y.float(),p=self.p) # L,B,B
        forward_scores = solve_relaxed_wmd(scores,dim=1)
        backward_scores = solve_relaxed_wmd(scores,dim=2)
        return torch.max(forward_scores,backward_scores)

def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res