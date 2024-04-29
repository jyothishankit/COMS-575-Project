import torch
import numpy as np
from torch import nn, einsum
from joblib import Parallel, delayed
from einops import rearrange
from einops.layers.torch import Rearrange
from tools.compute_metrics import compute_metrics


def kaiming_init(m):
    if isinstance(m, type(torch.nn.Linear())):
        _init_linear(m)
    elif isinstance(m, type(torch.nn.Conv2d(1, 1, 1))):
        _init_conv2d(m)
    elif isinstance(m, type(torch.nn.Conv1d(1, 1, 1))):
        _init_conv1d(m)

def _init_linear(m):
    torch.nn.init.kaiming_normal_(m.weight)
    if m.bias is not None:
        m.bias.data.fill_(0.01)

def _init_conv2d(m):
    torch.nn.init.kaiming_normal_(m.weight)
    if m.bias is not None:
        m.bias.data.fill_(0.01)

def _init_conv1d(m):
    torch.nn.init.kaiming_normal_(m.weight)
    if m.bias is not None:
        m.bias.data.fill_(0.01)



def power_compress(x):
    real, imag = extract_real_imag_parts(x)
    spec = create_complex_tensor(real, imag)
    mag = calculate_magnitude(spec)
    phase = calculate_phase(spec)
    mag = compress_magnitude(mag)
    real_compress, imag_compress = compress_real_imag(mag, phase)
    return combine_real_imag(real_compress, imag_compress)

def extract_real_imag_parts(x):
    real = x[..., 0]
    imag = x[..., 1]
    return real, imag

def create_complex_tensor(real, imag):
    return torch.complex(real, imag)

def calculate_magnitude(spec):
    return torch.abs(spec)

def calculate_phase(spec):
    return torch.angle(spec)

def compress_magnitude(mag):
    return mag ** 0.3

def compress_real_imag(mag, phase):
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return real_compress, imag_compress

def combine_real_imag(real_compress, imag_compress):
    return torch.stack([real_compress, imag_compress], 1)



def power_uncompress(real, imag):
    spec = create_complex_tensor(real, imag)
    mag = calculate_magnitude(spec)
    phase = calculate_phase(spec)
    mag = uncompress_magnitude(mag)
    real_uncompress, imag_uncompress = uncompress_real_imag(mag, phase)
    return combine_real_imag(real_uncompress, imag_uncompress)

def create_complex_tensor(real, imag):
    return torch.complex(real, imag)

def calculate_magnitude(spec):
    return torch.abs(spec)

def calculate_phase(spec):
    return torch.angle(spec)

def uncompress_magnitude(mag):
    return mag ** (1.0 / 0.3)

def uncompress_real_imag(mag, phase):
    real_uncompress = mag * torch.cos(phase)
    imag_uncompress = mag * torch.sin(phase)
    return real_uncompress, imag_uncompress

def combine_real_imag(real_uncompress, imag_uncompress):
    return torch.stack([real_uncompress, imag_uncompress], -1)



def exists(value):
    return value is not None


def default(value, default_value):
    return value if exists(value) else default_value


def calc_same_padding(kernel_size):
    padding_left = calculate_left_padding(kernel_size)
    padding_right = calculate_right_padding(kernel_size)
    return (padding_left, padding_right)

def calculate_left_padding(kernel_size):
    return kernel_size // 2

def calculate_right_padding(kernel_size):
    return kernel_size // 2 - (kernel_size + 1) % 2





class Swish(nn.Module):
    def forward(self, x):
        return self._swish_activation(x)

    def _swish_activation(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = self._chunk_input(x)
        return out * gate.sigmoid()

    def _chunk_input(self, x):
        return x.chunk(2, dim=self.dim)



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = self._initialize_layer_norm(dim)

    def forward(self, x, **kwargs):
        x_normalized = self.norm(x)
        return self.fn(x_normalized, **kwargs)

    def _initialize_layer_norm(self, dim):
        return nn.LayerNorm(dim)

    
class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = self._initialize_slope(in_features)

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

    def _initialize_slope(self, in_features):
        return nn.Parameter(torch.ones(in_features, requires_grad=True))


def metric_score_loss(clean, noisy, sr=16000):
    try:
        metrics = calculate_metrics(clean, noisy, sr)
        return combine_metrics(metrics)
    except Exception as e:
        print("Error in SSNR and STOI calculation:", e)

def calculate_metrics(clean, noisy, sr):
    try:
        metrics = compute_metrics(clean, noisy, sr, 0)
        return np.array(metrics)
    except Exception as e:
        raise e

def combine_metrics(metrics):
    return metrics[4] + metrics[5]



def batch_metric_score(clean, noisy):
    try:
        metric_scores = calculate_metric_scores(clean, noisy)
        return process_metric_scores(metric_scores)
    except Exception as e:
        print("Error in batch metric score computation:", e)
        return None

def calculate_metric_scores(clean, noisy):
    try:
        metric_score_scores = Parallel(n_jobs=-1)(
            delayed(metric_score_loss)(c, n) for c, n in zip(clean, noisy)
        )
        return np.array(metric_score_scores)
    except Exception as e:
        raise e

def process_metric_scores(metric_scores):
    if -1 in metric_scores:
        return None
    metric_scores = (metric_scores - 1) / 3.5
    return torch.FloatTensor(metric_scores).to("cuda")



class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.to_q = self._initialize_linear(dim, inner_dim)
        self.to_kv = self._initialize_linear(dim, inner_dim * 2)
        self.to_out = self._initialize_linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = self._initialize_positional_embedding(dim_head, max_pos_emb)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_context = (
            x.shape[-2],
            x.device,
            self.heads,
            self.max_pos_emb,
            exists(context),
        )
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: self._rearrange_for_heads(t, h), (q, k, v))

        dots = self._compute_scaled_dot_product(q, k, self.scale)

        rel_pos_emb = self._compute_relative_positional_embedding(n, max_pos_emb, q)
        pos_attn = self._compute_positional_attention(q, rel_pos_emb, self.scale)
        dots = dots + pos_attn

        self._apply_attention_mask(dots, mask, context_mask)

        attn = dots.softmax(dim=-1)
        out = self._apply_attention(attn, v)

        return self.dropout(out)

    def _initialize_linear(self, in_features, out_features):
        return nn.Linear(in_features, out_features, bias=False)

    def _initialize_positional_embedding(self, dim_head, max_pos_emb):
        return nn.Embedding(2 * max_pos_emb + 1, dim_head)

    def _rearrange_for_heads(self, tensor, heads):
        return rearrange(tensor, "b n (h d) -> b h n d", h=heads)

    def _compute_scaled_dot_product(self, q, k, scale):
        return einsum("b h i d, b h j d -> b h i j", q, k) * scale

    def _compute_relative_positional_embedding(self, n, max_pos_emb, q):
        seq = torch.arange(n, device=q.device)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        return self.rel_pos_emb(dist).to(q)

    def _compute_positional_attention(self, q, rel_pos_emb, scale):
        return einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * scale

    def _apply_attention_mask(self, dots, mask, context_mask):
        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*dots.shape[:2], device=dots.device))
            context_mask = (
                default(context_mask, mask)
                if not has_context
                else default(
                    context_mask, lambda: torch.ones(*context.shape[:2], device=dots.device)
                )
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, mask_value)

    def _apply_attention(self, attn, v):
        return einsum("b h i j, b h j d -> b h i d", attn, v).contiguous()

