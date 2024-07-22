import torch.nn.functional as F
import torch

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

    def forward(self,x,y):
        L, B, H = x.size()
        scores = torch.cdist(x,y,p=self.p)
        logits = scores / self.tau
        logits_max, _ = torch.max(logits,dim=1,keepdim=True)
        logits = logits - logits_max.detach()
        logprob = logits - logits.logsumexp(dim=-1,keepdim=True)
        loss = - logprob[:,range(B),range(B)]
        return loss.mean()

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