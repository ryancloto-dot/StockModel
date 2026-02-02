# Stockmodel.py
# ------------------------------------------------------------
# Cognitive Stock Model (Agents -> Fast -> Medium Gate -> Slow)
# With IMPORTANT fixes:
#   (1) Medium learns epistemics (Fast entropy + agent disagreement + fast-agent disagreement)
#       and is supervised by "Fast is wrong" (top-quantile fast loss), not a hand proxy.
#   (2) Slow is corrective: delta objective vs Fast (only rewarded when it improves Fast).
#   (3) Transaction cost pressure: discourage non-HOLD probability (simple, minimal churn proxy).
#
# Works on Windows WITHOUT Triton (torch.compile auto-disables if Triton missing).
# Intended for Linux + L40S/A100 too.
# ------------------------------------------------------------
import os

# ---- TorchInductor / Triton cleanup ----
os.environ["TORCH_LOGS"] = ""
os.environ["TORCHDYNAMO_DISABLE"] = "0"
os.environ["TORCHINDUCTOR_VERBOSE"] = "0"
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/workspace/torchinductor_cache"

import os
import json
import time
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import os

# --- hard reset torch logging state (prevents dict corruption) ---
for k in list(os.environ.keys()):
    if k.startswith("TORCH_"):
        if "LOG" in k or "DYNAMO" in k or "INDUCTOR" in k:
            os.environ.pop(k, None)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    dataset: str = "synthetic"   # synthetic | real_csv (placeholder)
    data_dir: str = "data_real"

    # sampling
    n_samples: int = 20000
    n_stocks: int = 500
    total_days: int = 2500
    lookback: int = 128
    horizon: int = 30

    # fake web
    vocab_size: int = 4096
    text_len: int = 192
    pages_per_agent: int = 4
    hallucination_rate: float = 0.2
    n_agents: int = 8

    # loader
    batch_size: int = 64
    num_workers: int = 0

    # stage epochs
    agent_epochs: int = 1
    fast_epochs: int = 2
    medium_epochs: int = 2
    slow_epochs: int = 6

    # optim
    lr_agents: float = 3e-4
    lr_fast: float = 2e-4
    lr_medium: float = 2e-4
    lr_slow: float = 2e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    # mixed precision / perf
    amp: bool = False
    amp_dtype: str = "bf16"   # bf16 | fp16
    allow_tf32: bool = True
    compile: bool = False

    # model dims
    d_lat: int = 128
    d_model: int = 1024
    n_layers_fast: int = 10
    n_heads_fast: int = 16
    ff_mult: int = 4
    dropout: float = 0.1

    # slow block
    slow_tokens: int = 16
    n_layers_slow: int = 4
    n_heads_slow: int = 16

    # losses
    w_returns: float = 1.0
    w_action_day: float = 0.35
    w_action_sum: float = 0.5
    w_entropy: float = 0.02
    label_smoothing: float = 0.05

    # NEW: delta objective for Slow (corrective)
    w_slow_delta: float = 0.5

    # NEW: transaction-cost proxy (discourage non-HOLD probability)
    # Keep small; it's a "minimal pressure" knob.
    w_trade_smooth: float = 0.05
    hold_index: int = 1  # classes: 0=SELL, 1=HOLD, 2=BUY

    # gate behavior
    slow_threshold: float = 0.14        # hard threshold for routing in stage 3
    slow_target_rate: float = 0.15      # target fraction to route (calibration)
    gate_cal_w: float = 0.2             # weight on calibration penalty

    # io
    out_dir: str = "./checkpoints_stock_cognitive_l40s"
    dump_every_epochs: int = 1
    dump_k_samples: int = 250
    save_every: int = 1

    # resume
    resume_agents: str = ""
    resume_fast: str = ""
    resume_medium: str = ""
    resume_slow: str = ""


# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now():
    return time.time()

def _try_load_state_dict(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        # assume it's already a state_dict
        return obj
    raise ValueError(f"Unknown checkpoint format: {path}")

def save_state_dict(path: str, module: nn.Module):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(module.state_dict(), path)

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def human_m(n: int) -> str:
    return f"{n/1e6:.2f}M"

def get_autocast_dtype(cfg: Config):
    if cfg.amp_dtype.lower() == "bf16":
        return torch.bfloat16
    if cfg.amp_dtype.lower() == "fp16":
        return torch.float16
    raise ValueError("amp_dtype must be bf16 or fp16")

def maybe_enable_perf(cfg: Config, device: torch.device):
    if device.type == "cuda":
        if cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

def maybe_compile(cfg: Config, module: nn.Module) -> nn.Module:
    if not cfg.compile:
        return module

    try:
        import triton  # noqa: F401
    except Exception:
        print("[compile] Triton not available -> disabling torch.compile.")
        return module

    try:
        import torch
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True

        torch._inductor.config.max_autotune = False
        torch._inductor.config.triton.autotune = False
        torch._inductor.config.triton.cudagraphs = True

        return torch.compile(module, mode="reduce-overhead")

    except Exception as e:
        print(f"[compile] torch.compile failed -> fallback to eager. Reason: {e}")
        return module


# ----------------------------
# Synthetic Dataset
# ----------------------------

def _make_fake_web(cfg: Config, B: int, device: torch.device) -> torch.Tensor:
    # tokens: [B, A, P, T]
    # We'll generate random ids with optional "hallucination" corruption
    A, P, T = cfg.n_agents, cfg.pages_per_agent, cfg.text_len
    x = torch.randint(0, cfg.vocab_size, (B, A, P, T), device=device)
    if cfg.hallucination_rate > 0:
        mask = torch.rand_like(x.float()) < cfg.hallucination_rate
        noise = torch.randint(0, cfg.vocab_size, x.shape, device=device)
        x = torch.where(mask, noise, x)
    return x

def _price_to_features(price: torch.Tensor) -> torch.Tensor:
    # price: [B, L] (positive)
    # return features: [B, L, 2] => log_return, zscore(price)
    lr = torch.log(price[:, 1:] / price[:, :-1]).clamp(-0.2, 0.2)  # [B, L-1]
    lr = F.pad(lr, (1, 0), value=0.0)  # [B, L]
    mean = price.mean(dim=1, keepdim=True)
    std = price.std(dim=1, keepdim=True).clamp_min(1e-6)
    zp = (price - mean) / std
    return torch.stack([lr, zp], dim=-1)

def _actions_from_returns(r: torch.Tensor, thr: float = 0.002) -> torch.Tensor:
    # r: [B] signed return
    # 0=SELL, 1=HOLD, 2=BUY
    a = torch.ones_like(r, dtype=torch.long)
    a = torch.where(r > thr, torch.full_like(a, 2), a)
    a = torch.where(r < -thr, torch.full_like(a, 0), a)
    return a

class SyntheticStockDataset(Dataset):
    def __init__(self, cfg: Config, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.N = cfg.n_samples
        # deterministic split by offset
        self.offset = 0 if split == "train" else 1234567

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        cfg = self.cfg
        # stable RNG per index for reproducibility
        g = torch.Generator().manual_seed(cfg.seed + self.offset + idx)

        L = cfg.lookback + cfg.horizon + 1
        # random-walk-ish process with regime shifts
        vol = torch.rand((), generator=g).item() * 0.02 + 0.002  # ~[0.2%, 2.2%]
        drift = (torch.rand((), generator=g).item() - 0.5) * 0.001
        shocks = torch.randn(L, generator=g) * vol + drift

        # create prices
        logp = shocks.cumsum(dim=0)
        price = torch.exp(logp).clamp_min(1e-4)

        # split
        price_lb = price[:cfg.lookback + 1]         # [lookback+1]
        price_future = price[cfg.lookback + 1:]     # [horizon]

        # features
        x_price = _price_to_features(price_lb.unsqueeze(0)).squeeze(0)  # [lookback+1, 2]
        x_price = x_price[1:]  # align to [lookback,2]

        # targets: future log returns for horizon
        y_returns = torch.log(price_future / price[cfg.lookback:cfg.lookback + cfg.horizon]).clamp(-0.2, 0.2)
        # day action based on day-1 return
        y_day = _actions_from_returns(y_returns[0])
        # sum action based on sum horizon
        y_sum = _actions_from_returns(y_returns.sum())

        # web tokens produced later on device (for speed), but return a placeholder
        # We'll return None and generate in collate_fn.
        return {
            "x_price": x_price.float(),          # [L,2]
            "y_returns": y_returns.float(),      # [H]
            "y_day": y_day.long(),               # []
            "y_sum": y_sum.long(),               # []
        }

def collate_synth(cfg: Config, batch, device: torch.device):
    x_price = torch.stack([b["x_price"] for b in batch], dim=0).to(device)         # [B,L,2]
    y_returns = torch.stack([b["y_returns"] for b in batch], dim=0).to(device)     # [B,H]
    y_day = torch.stack([b["y_day"] for b in batch], dim=0).to(device)             # [B]
    y_sum = torch.stack([b["y_sum"] for b in batch], dim=0).to(device)             # [B]
    web = _make_fake_web(cfg, x_price.size(0), device)                              # [B,A,P,T]
    return {"x_price": x_price, "web": web, "y_returns": y_returns, "y_day": y_day, "y_sum": y_sum}


# ----------------------------
# Model Blocks
# ----------------------------

class Agents(nn.Module):
    """
    FakeWeb -> latent evidence per agent
    Input: web [B, A, P, T]
    Output:
      z_all [B, A, d_lat]
      aux_logits [B, A, 3]  (optional auxiliary classification head)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d_model = min(cfg.d_model, 512)  # keep agents lightweight
        self.emb = nn.Embedding(cfg.vocab_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.proj = nn.Linear(d_model, cfg.d_lat)

        self.aux = nn.Sequential(
            nn.Linear(cfg.d_lat, cfg.d_lat),
            nn.GELU(),
            nn.Linear(cfg.d_lat, 3),
        )

    def forward(self, web: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, A, P, T = web.shape
        x = web.view(B * A * P, T)              # [B*A*P, T]
        h = self.emb(x)                         # [B*A*P, T, d_model]
        h = self.enc(h)                         # [B*A*P, T, d_model]
        h = h.mean(dim=1)                       # [B*A*P, d_model]
        z = self.proj(h)                        # [B*A*P, d_lat]
        z = z.view(B, A, P, self.cfg.d_lat).mean(dim=2)  # [B,A,d_lat]
        aux = self.aux(z)                       # [B,A,3]
        return z, aux


class FastModel(nn.Module):
    """
    Always-on transformer over price features
    Input: x_price [B, L, 2]
    Output: dict with:
      fast_summary [B, d_lat]
      returns [B, H]
      logits_day [B,3]
      logits_sum [B,3]
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(2, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads_fast,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers_fast)
        self.to_lat = nn.Linear(cfg.d_model, cfg.d_lat)

        self.head_returns = nn.Linear(cfg.d_lat, cfg.horizon)
        self.head_day = nn.Linear(cfg.d_lat, 3)
        self.head_sum = nn.Linear(cfg.d_lat, 3)

    def forward(self, x_price: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.inp(x_price)                   # [B,L,d_model]
        h = self.enc(h)                         # [B,L,d_model]
        pooled = h.mean(dim=1)                  # [B,d_model]
        fast_summary = self.to_lat(pooled)      # [B,d_lat]
        returns = self.head_returns(fast_summary)  # [B,H]
        logits_day = self.head_day(fast_summary)   # [B,3]
        logits_sum = self.head_sum(fast_summary)   # [B,3]
        return {
            "fast_summary": fast_summary,
            "returns": returns,
            "logits_day": logits_day,
            "logits_sum": logits_sum,
        }


class MediumGate(nn.Module):
    """
    Epistemic gate.
    Takes:
      fast_summary [B,d_lat]
      agent_mean   [B,d_lat]
      stats        [B, 9]  (6 price stats + entropy_fast + agent_var + fast_agent_disagreement)
    Outputs:
      p_slow [B] in [0,1]
      gate_logit [B]
    """
    def __init__(self, cfg: Config, stats_dim: int = 9):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.d_lat + cfg.d_lat + stats_dim
        hid = max(256, cfg.d_lat * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, 1),
        )

    def forward(self, fast_summary: torch.Tensor, agent_mean: torch.Tensor, stats: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([fast_summary, agent_mean, stats], dim=-1)
        logit = self.net(x).squeeze(-1)
        p = torch.sigmoid(logit)
        return {"p_slow": p, "gate_logit": logit}


class SlowModel(nn.Module):
    """
    Rare, expensive corrective block.
    Input: fast_summary [B,d_lat]
    We expand into slow_tokens and run a small transformer.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_proj = nn.Linear(cfg.d_lat, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(cfg.slow_tokens, cfg.d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads_slow,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers_slow)
        self.to_lat = nn.Linear(cfg.d_model, cfg.d_lat)

        self.head_returns = nn.Linear(cfg.d_lat, cfg.horizon)
        self.head_day = nn.Linear(cfg.d_lat, 3)
        self.head_sum = nn.Linear(cfg.d_lat, 3)

    def forward(self, fast_summary: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = fast_summary.size(0)
        t = self.token_proj(fast_summary).unsqueeze(1).repeat(1, self.cfg.slow_tokens, 1)  # [B,T,d_model]
        t = t + self.pos.unsqueeze(0)                                                     # [B,T,d_model]
        h = self.enc(t)
        pooled = h.mean(dim=1)
        slow_summary = self.to_lat(pooled)

        returns = self.head_returns(slow_summary)
        logits_day = self.head_day(slow_summary)
        logits_sum = self.head_sum(slow_summary)

        return {
            "slow_summary": slow_summary,
            "returns": returns,
            "logits_day": logits_day,
            "logits_sum": logits_sum,
        }


class CognitiveStockModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.agents = Agents(cfg)
        self.fast = FastModel(cfg)
        self.medium = MediumGate(cfg, stats_dim=9)
        self.slow = SlowModel(cfg)

    @staticmethod
    def stats_from_price(x_price: torch.Tensor) -> torch.Tensor:
        """
        x_price: [B, L, 2] where feat0 is log_return, feat1 is zscore price
        returns stats: [B, 6]
        """
        lr = x_price[:, :, 0]
        zp = x_price[:, :, 1]
        mean_lr = lr.mean(dim=1)
        std_lr = lr.std(dim=1)
        max_abs_lr = lr.abs().max(dim=1).values
        mean_zp = zp.mean(dim=1)
        std_zp = zp.std(dim=1)
        max_abs_zp = zp.abs().max(dim=1).values
        return torch.stack([mean_lr, std_lr, max_abs_lr, mean_zp, std_zp, max_abs_zp], dim=-1)

    @staticmethod
    def fast_day_entropy(logits_day: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits_day, dim=-1)
        ent = -(probs * (probs + 1e-9).log()).sum(dim=-1)
        return ent  # [B]

    @staticmethod
    def agent_disagreement(z_all: torch.Tensor) -> torch.Tensor:
        # z_all: [B,A,d_lat]
        # return scalar disagreement per sample: mean var across agents and dims
        var = z_all.var(dim=1, unbiased=False).mean(dim=-1)  # [B]
        return var

    @staticmethod
    def fast_agent_disagreement(fast_summary: torch.Tensor, agent_mean: torch.Tensor) -> torch.Tensor:
        # 1 - cosine similarity
        cs = F.cosine_similarity(fast_summary, agent_mean, dim=-1).clamp(-1, 1)
        return (1.0 - cs)  # [B]


# ----------------------------
# Losses / Metrics
# ----------------------------

def label_smoothed_ce_per_sample(logits: torch.Tensor, targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """
    logits: [B, C], targets: [B]
    returns per-sample CE: [B]
    """
    if smoothing <= 0:
        return F.cross_entropy(logits, targets, reduction="none")
    C = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(dim=-1)
    return (1.0 - smoothing) * nll + smoothing * smooth

def compute_task_loss_per_sample(cfg: Config,
                                 pred_returns: torch.Tensor, logits_day: torch.Tensor, logits_sum: torch.Tensor,
                                 y_returns: torch.Tensor, y_day: torch.Tensor, y_sum: torch.Tensor) -> torch.Tensor:
    # mse per sample
    mse = ((pred_returns - y_returns) ** 2).mean(dim=-1)  # [B]
    ce_day = label_smoothed_ce_per_sample(logits_day, y_day, cfg.label_smoothing)  # [B]
    ce_sum = label_smoothed_ce_per_sample(logits_sum, y_sum, cfg.label_smoothing)  # [B]
    # entropy bonus (negative in loss) -> we'll keep batch-level in scalar loss only
    loss = cfg.w_returns * mse + cfg.w_action_day * ce_day + cfg.w_action_sum * ce_sum
    return loss  # [B]

def compute_task_loss(cfg: Config,
                      pred_returns: torch.Tensor, logits_day: torch.Tensor, logits_sum: torch.Tensor,
                      y_returns: torch.Tensor, y_day: torch.Tensor, y_sum: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    per = compute_task_loss_per_sample(cfg, pred_returns, logits_day, logits_sum, y_returns, y_day, y_sum)
    base = per.mean()

    # small entropy bonus to avoid collapse
    p_day = F.softmax(logits_day, dim=-1)
    p_sum = F.softmax(logits_sum, dim=-1)
    ent = -(p_day * (p_day.clamp_min(1e-9).log())).sum(dim=-1).mean()
    ent += -(p_sum * (p_sum.clamp_min(1e-9).log())).sum(dim=-1).mean()
    ent = ent * 0.5

    # transaction-cost proxy: encourage HOLD probability (minimal churn pressure)
    p_hold = p_day[:, cfg.hold_index]
    trade_pen = (1.0 - p_hold).mean()

    loss = base - cfg.w_entropy * ent + cfg.w_trade_smooth * trade_pen

    stats = {
        "loss": float(loss.item()),
        "base": float(base.item()),
        "ent": float(ent.item()),
        "trade_pen": float(trade_pen.item()),
    }
    return loss, stats

@torch.no_grad()
def compute_metrics(pred_returns: torch.Tensor, logits_day: torch.Tensor, logits_sum: torch.Tensor,
                    y_returns: torch.Tensor, y_day: torch.Tensor, y_sum: torch.Tensor) -> Dict[str, float]:
    mse = float(F.mse_loss(pred_returns, y_returns).item())
    pred_day = torch.argmax(logits_day, dim=-1)
    pred_sum = torch.argmax(logits_sum, dim=-1)
    acc = float(((pred_day == y_day) & (pred_sum == y_sum)).float().mean().item())

    true_sign = torch.sign(y_returns.sum(dim=-1))
    pred_sign = torch.sign(pred_returns.sum(dim=-1))
    sign_acc = float((true_sign == pred_sign).float().mean().item())
    return {"mse": mse, "acc": acc, "sign": sign_acc}


# ----------------------------
# Training Stages
# ----------------------------

def _make_dataloaders(cfg: Config, device: torch.device):
    if cfg.dataset != "synthetic":
        raise NotImplementedError("real_csv loader not implemented in this minimal script.")

    train_ds = SyntheticStockDataset(cfg, split="train")
    val_ds = SyntheticStockDataset(cfg, split="val")

    def _collate(batch):
        return collate_synth(cfg, batch, device)

    # safer on Windows: num_workers=0
    nw = cfg.num_workers
    if os.name == "nt" and nw != 0:
        print("[note] Windows detected: forcing --num_workers 0 (safer).")
        nw = 0

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=nw, collate_fn=_collate, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=nw, collate_fn=_collate, pin_memory=False)
    return train_dl, val_dl

def _make_scaler(cfg: Config, device: torch.device):
    enabled = (cfg.amp and device.type == "cuda")
    # new API
    return torch.amp.GradScaler("cuda", enabled=enabled)

def _autocast_ctx(cfg: Config, device: torch.device):
    if not (cfg.amp and device.type == "cuda"):
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=get_autocast_dtype(cfg), enabled=True)

@torch.no_grad()
def eval_fast(cfg: Config, model: CognitiveStockModel, dl: DataLoader, device: torch.device, max_batches: int = 50):
    model.eval()
    m = {"mse": 0.0, "acc": 0.0, "sign": 0.0}
    n = 0
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        x_price = batch["x_price"]
        y_returns = batch["y_returns"]
        y_day = batch["y_day"]
        y_sum = batch["y_sum"]

        out = model.fast(x_price)
        met = compute_metrics(out["returns"], out["logits_day"], out["logits_sum"], y_returns, y_day, y_sum)
        for k in m:
            m[k] += met[k]
        n += 1
    for k in m:
        m[k] /= max(n, 1)
    model.train()
    return m

def train_agents(cfg: Config, model: CognitiveStockModel, train_dl: DataLoader, device: torch.device):
    opt = torch.optim.AdamW(model.agents.parameters(), lr=cfg.lr_agents, weight_decay=cfg.weight_decay)
    scaler = _make_scaler(cfg, device)

    for ep in range(cfg.agent_epochs):
        t0 = now()
        loss_sum = 0.0
        steps = 0
        for batch in train_dl:
            web = batch["web"]

            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(cfg, device):
                z_all, aux_logits = model.agents(web)
                # simple auxiliary self-supervision: predict synthetic "stance" from agent mean token id statistic
                # We create a fake target from web content so agents learn something non-trivial.
                # target: 0/1/2 based on mean token id bucket
                B = web.size(0)
                token_mean = web.float().mean(dim=(2, 3))  # [B,A]
                y_aux = torch.bucketize(token_mean, boundaries=torch.tensor([cfg.vocab_size/3, 2*cfg.vocab_size/3], device=token_mean.device)).long()
                loss_aux = F.cross_entropy(aux_logits.view(B * cfg.n_agents, 3), y_aux.view(B * cfg.n_agents))
                # small penalty to keep latents bounded
                loss = loss_aux + 0.001 * (z_all ** 2).mean()

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.agents.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.item())
            steps += 1

        print(f"[Agents] Epoch {ep+1}/{cfg.agent_epochs} loss={loss_sum/max(steps,1):.4f} time={now()-t0:.1f}s")

def train_fast(cfg: Config, model: CognitiveStockModel, train_dl: DataLoader, val_dl: DataLoader, device: torch.device):
    opt = torch.optim.AdamW(model.fast.parameters(), lr=cfg.lr_fast, weight_decay=cfg.weight_decay)
    scaler = _make_scaler(cfg, device)

    for ep in range(cfg.fast_epochs):
        t0 = now()
        loss_sum = 0.0
        steps = 0
        for batch in train_dl:
            x_price = batch["x_price"]
            y_returns = batch["y_returns"]
            y_day = batch["y_day"]
            y_sum = batch["y_sum"]

            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(cfg, device):
                out = model.fast(x_price)
                loss, _ = compute_task_loss(cfg, out["returns"], out["logits_day"], out["logits_sum"], y_returns, y_day, y_sum)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.fast.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.item())
            steps += 1

        met = eval_fast(cfg, model, val_dl, device, max_batches=50)
        print(f"[Fast] Epoch {ep+1}/{cfg.fast_epochs} loss={loss_sum/max(steps,1):.4f} "
              f"val_mse={met['mse']:.6f} acc={met['acc']*100:.2f}% sign={met['sign']*100:.2f}% time={now()-t0:.1f}s")

def train_medium(cfg: Config, model: CognitiveStockModel, train_dl: DataLoader, val_dl: DataLoader, device: torch.device):
    """
    NEW: Medium is trained to predict "Fast likely wrong" using epistemic signals.
    Label construction per batch:
      fast_loss_per_sample -> label = 1 if in top-quantile (target rate = slow_target_rate)
    Inputs include:
      price stats (6)
      entropy_fast (1)
      agent disagreement var (1)
      fast-agent disagreement (1)
    Objective:
      BCE(gate_logit, label) + gate_cal_w*(mean(p_slow)-target)^2
    """
    opt = torch.optim.AdamW(model.medium.parameters(), lr=cfg.lr_medium, weight_decay=cfg.weight_decay)
    scaler = _make_scaler(cfg, device)

    def _gate_inputs(x_price, web):
        # agents
        z_all, _ = model.agents(web)               # [B,A,d_lat]
        agent_mean = z_all.mean(dim=1)             # [B,d_lat]
        agent_var = CognitiveStockModel.agent_disagreement(z_all)  # [B]
        # fast
        out_fast = model.fast(x_price)
        fast_summary = out_fast["fast_summary"]    # [B,d_lat]
        ent_fast = CognitiveStockModel.fast_day_entropy(out_fast["logits_day"])  # [B]
        fa_dis = CognitiveStockModel.fast_agent_disagreement(fast_summary, agent_mean)  # [B]
        # stats
        stats6 = CognitiveStockModel.stats_from_price(x_price)  # [B,6]
        stats = torch.cat([
            stats6,
            ent_fast.unsqueeze(-1),
            agent_var.unsqueeze(-1),
            fa_dis.unsqueeze(-1),
        ], dim=-1)  # [B,9]
        return out_fast, z_all, agent_mean, stats

    for ep in range(cfg.medium_epochs):
        t0 = now()
        loss_sum = 0.0
        steps = 0
        gate_rate_sum = 0.0

        for batch in train_dl:
            x_price = batch["x_price"]
            web = batch["web"]
            y_returns = batch["y_returns"]
            y_day = batch["y_day"]
            y_sum = batch["y_sum"]

            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                out_fast, z_all, agent_mean, stats = _gate_inputs(x_price, web)
                fast_loss_per = compute_task_loss_per_sample(
                    cfg, out_fast["returns"], out_fast["logits_day"], out_fast["logits_sum"],
                    y_returns, y_day, y_sum
                )  # [B]
                # target threshold by quantile to match slow_target_rate
                q = max(0.0, min(1.0, 1.0 - cfg.slow_target_rate))
                thr = torch.quantile(fast_loss_per, q=q)
                y_need = (fast_loss_per >= thr).float()  # [B]

            with _autocast_ctx(cfg, device):
                gate_out = model.medium(out_fast["fast_summary"].detach(), agent_mean.detach(), stats.detach())
                p = gate_out["p_slow"]
                logit = gate_out["gate_logit"]

                bce = F.binary_cross_entropy_with_logits(logit, y_need)
                cal = (p.mean() - cfg.slow_target_rate) ** 2
                loss = bce + cfg.gate_cal_w * cal

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.medium.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.item())
            gate_rate_sum += float((p.detach() >= cfg.slow_threshold).float().mean().item())
            steps += 1

        # quick val rate check
        with torch.no_grad():
            model.eval()
            rates = []
            bces = []
            for i, batch in enumerate(val_dl):
                if i >= 30:
                    break
                x_price = batch["x_price"]
                web = batch["web"]
                y_returns = batch["y_returns"]
                y_day = batch["y_day"]
                y_sum = batch["y_sum"]

                out_fast, z_all, agent_mean, stats = _gate_inputs(x_price, web)
                fast_loss_per = compute_task_loss_per_sample(
                    cfg, out_fast["returns"], out_fast["logits_day"], out_fast["logits_sum"],
                    y_returns, y_day, y_sum
                )
                q = max(0.0, min(1.0, 1.0 - cfg.slow_target_rate))
                thr = torch.quantile(fast_loss_per, q=q)
                y_need = (fast_loss_per >= thr).float()

                gate_out = model.medium(out_fast["fast_summary"], agent_mean, stats)
                p = gate_out["p_slow"]
                bce = F.binary_cross_entropy_with_logits(gate_out["gate_logit"], y_need)
                rates.append(float((p >= cfg.slow_threshold).float().mean().item()))
                bces.append(float(bce.item()))
            model.train()
        print(f"[Medium] Epoch {ep+1}/{cfg.medium_epochs} "
              f"loss={loss_sum/max(steps,1):.4f} val_bce={sum(bces)/max(len(bces),1):.4f} "
              f"gate_rate={gate_rate_sum/max(steps,1):.3f} val_rate={sum(rates)/max(len(rates),1):.3f} time={now()-t0:.1f}s")

def train_slow(cfg: Config, model: CognitiveStockModel, train_dl: DataLoader, val_dl: DataLoader, device: torch.device):
    """
    NEW: Slow is corrective:
      loss = loss_slow + w_slow_delta * clamp(loss_slow - loss_fast_detached, min=0)
    Routing:
      use Medium p_slow >= slow_threshold to select routed samples
    """
    opt = torch.optim.AdamW(model.slow.parameters(), lr=cfg.lr_slow, weight_decay=cfg.weight_decay)
    scaler = _make_scaler(cfg, device)

    def _gate_and_fast(x_price, web):
        z_all, _ = model.agents(web)
        agent_mean = z_all.mean(dim=1)
        agent_var = CognitiveStockModel.agent_disagreement(z_all)
        out_fast = model.fast(x_price)
        ent_fast = CognitiveStockModel.fast_day_entropy(out_fast["logits_day"])
        fa_dis = CognitiveStockModel.fast_agent_disagreement(out_fast["fast_summary"], agent_mean)
        stats6 = CognitiveStockModel.stats_from_price(x_price)
        stats = torch.cat([stats6, ent_fast.unsqueeze(-1), agent_var.unsqueeze(-1), fa_dis.unsqueeze(-1)], dim=-1)
        gate_out = model.medium(out_fast["fast_summary"].detach(), agent_mean.detach(), stats.detach())
        p_slow = gate_out["p_slow"]
        return out_fast, p_slow

    for ep in range(cfg.slow_epochs):
        t0 = now()
        loss_sum = 0.0
        steps = 0
        routed_sum = 0.0

        for batch in train_dl:
            x_price = batch["x_price"]
            web = batch["web"]
            y_returns = batch["y_returns"]
            y_day = batch["y_day"]
            y_sum = batch["y_sum"]

            with torch.no_grad():
                out_fast, p_slow = _gate_and_fast(x_price, web)
                mask = (p_slow >= cfg.slow_threshold)
                routed_rate = float(mask.float().mean().item())

            if mask.sum().item() == 0:
                # nothing routed this batch; skip update to avoid shape issues
                routed_sum += routed_rate
                steps += 1
                continue

            x_fs = out_fast["fast_summary"][mask]       # [Br, d_lat]
            yR = y_returns[mask]
            yD = y_day[mask]
            yS = y_sum[mask]

            # fast loss per sample (detached)
            with torch.no_grad():
                fast_loss_per = compute_task_loss_per_sample(
                    cfg,
                    out_fast["returns"][mask],
                    out_fast["logits_day"][mask],
                    out_fast["logits_sum"][mask],
                    yR, yD, yS
                )  # [Br]
                fast_loss_det = fast_loss_per.mean()     # scalar

            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(cfg, device):
                out_slow = model.slow(x_fs)
                slow_loss, _ = compute_task_loss(cfg, out_slow["returns"], out_slow["logits_day"], out_slow["logits_sum"], yR, yD, yS)

                # corrective delta objective
                delta = (slow_loss - fast_loss_det).clamp(min=0.0)
                loss = slow_loss + cfg.w_slow_delta * delta

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.slow.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.item())
            routed_sum += routed_rate
            steps += 1

        # quick val: route + evaluate slow only on routed
        with torch.no_grad():
            model.eval()
            gate_rates = []
            mses = []
            accs = []
            signs = []
            for i, batch in enumerate(val_dl):
                if i >= 30:
                    break
                x_price = batch["x_price"]
                web = batch["web"]
                y_returns = batch["y_returns"]
                y_day = batch["y_day"]
                y_sum = batch["y_sum"]

                out_fast, p_slow = _gate_and_fast(x_price, web)
                mask = (p_slow >= cfg.slow_threshold)
                gate_rates.append(float(mask.float().mean().item()))
                if mask.sum().item() == 0:
                    continue

                out_slow = model.slow(out_fast["fast_summary"][mask])
                met = compute_metrics(out_slow["returns"], out_slow["logits_day"], out_slow["logits_sum"],
                                      y_returns[mask], y_day[mask], y_sum[mask])
                mses.append(met["mse"])
                accs.append(met["acc"])
                signs.append(met["sign"])
            model.train()

        val_mse = sum(mses)/max(len(mses), 1)
        val_acc = sum(accs)/max(len(accs), 1)
        val_sign = sum(signs)/max(len(signs), 1)
        gate_rate_val = sum(gate_rates)/max(len(gate_rates), 1)

        print(f"[Slow] Epoch {ep+1}/{cfg.slow_epochs} "
              f"loss={loss_sum/max(steps,1):.4f} routed_train={routed_sum/max(steps,1):.3f} "
              f"gate_rate_val={gate_rate_val:.3f} val_mse={val_mse:.6f} acc={val_acc*100:.2f}% sign={val_sign*100:.2f}% "
              f"time={now()-t0:.1f}s")


# ----------------------------
# Main
# ----------------------------

def build_argparser():
    p = argparse.ArgumentParser()
    # core
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dataset", type=str, default="synthetic")
    p.add_argument("--data_dir", type=str, default="data_real")

    # sampling
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--n_stocks", type=int, default=500)
    p.add_argument("--total_days", type=int, default=2500)
    p.add_argument("--lookback", type=int, default=128)
    p.add_argument("--horizon", type=int, default=30)

    # fake web
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--text_len", type=int, default=192)
    p.add_argument("--pages_per_agent", type=int, default=4)
    p.add_argument("--hallucination_rate", type=float, default=0.2)
    p.add_argument("--n_agents", type=int, default=8)

    # loader
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)

    # epochs
    p.add_argument("--agent_epochs", type=int, default=1)
    p.add_argument("--fast_epochs", type=int, default=2)
    p.add_argument("--medium_epochs", type=int, default=2)
    p.add_argument("--slow_epochs", type=int, default=6)

    # optim
    p.add_argument("--lr_agents", type=float, default=3e-4)
    p.add_argument("--lr_fast", type=float, default=2e-4)
    p.add_argument("--lr_medium", type=float, default=2e-4)
    p.add_argument("--lr_slow", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # perf
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="bf16")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--compile", action="store_true")

    # model dims
    p.add_argument("--d_lat", type=int, default=128)
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_layers_fast", type=int, default=10)
    p.add_argument("--n_heads_fast", type=int, default=16)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    # slow
    p.add_argument("--slow_tokens", type=int, default=16)
    p.add_argument("--n_layers_slow", type=int, default=4)
    p.add_argument("--n_heads_slow", type=int, default=16)

    # losses
    p.add_argument("--w_returns", type=float, default=1.0)
    p.add_argument("--w_action_day", type=float, default=0.35)
    p.add_argument("--w_action_sum", type=float, default=0.5)
    p.add_argument("--w_entropy", type=float, default=0.02)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--w_slow_delta", type=float, default=0.5)
    p.add_argument("--w_trade_smooth", type=float, default=0.05)
    p.add_argument("--hold_index", type=int, default=1)

    # gate
    p.add_argument("--slow_threshold", type=float, default=0.14)
    p.add_argument("--slow_target_rate", type=float, default=0.15)
    p.add_argument("--gate_cal_w", type=float, default=0.2)

    # io
    p.add_argument("--out_dir", type=str, default="./checkpoints_stock_cognitive_l40s")
    p.add_argument("--dump_every_epochs", type=int, default=1)
    p.add_argument("--dump_k_samples", type=int, default=250)
    p.add_argument("--save_every", type=int, default=1)

    # resume
    p.add_argument("--resume_agents", type=str, default="")
    p.add_argument("--resume_fast", type=str, default="")
    p.add_argument("--resume_medium", type=str, default="")
    p.add_argument("--resume_slow", type=str, default="")

    return p

def args_to_cfg(args) -> Config:
    cfg = Config()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg

def main():
    import subprocess
    from datetime import datetime

    def git_backup_checkpoints(out_dir: str):
        try:
            subprocess.run(["git", "add", out_dir], check=True)
            subprocess.run(["git", "add", ".gitattributes"], check=True)

            msg = f"backup checkpoints {datetime.utcnow().isoformat()}Z"
            subprocess.run(["git", "commit", "-m", msg], check=True)
            subprocess.run(["git", "push"], check=True)

            print("[backup] checkpoints pushed to GitHub")
        except subprocess.CalledProcessError as e:
            print(f"[backup] Git command failed: {e}")
        except Exception as e:
            print(f"[backup] Unexpected error: {e}")

    # ----------------------------
    # Setup
    # ----------------------------
    args = build_argparser().parse_args()
    cfg = args_to_cfg(args)

    set_seed(cfg.seed)

    device = torch.device(
        cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    cfg.device = str(device)

    maybe_enable_perf(cfg, device)

    if os.name == "nt":
        print("[note] Windows detected. If DataLoader hangs, use --num_workers 0")

    print("Config:", json.dumps(asdict(cfg), indent=2))
    print("Device:", device)

    train_dl, val_dl = _make_dataloaders(cfg, device)

    model = CognitiveStockModel(cfg).to(device)

    # ----------------------------
    # Optional compilation
    # ----------------------------
    if cfg.compile:
        print("[compile] torch.compile enabled (auto-fallback if Triton missing)")
        model.agents = maybe_compile(cfg, model.agents)
        model.fast = maybe_compile(cfg, model.fast)
        model.medium = maybe_compile(cfg, model.medium)
        model.slow = maybe_compile(cfg, model.slow)

    # ----------------------------
    # Resume checkpoints (optional)
    # ----------------------------
    if cfg.resume_agents:
        model.agents.load_state_dict(_try_load_state_dict(cfg.resume_agents))
    if cfg.resume_fast:
        model.fast.load_state_dict(_try_load_state_dict(cfg.resume_fast))
    if cfg.resume_medium:
        model.medium.load_state_dict(_try_load_state_dict(cfg.resume_medium))
    if cfg.resume_slow:
        model.slow.load_state_dict(_try_load_state_dict(cfg.resume_slow))

    # ----------------------------
    # Params
    # ----------------------------
    p_agents = count_params(model.agents)
    p_fast = count_params(model.fast)
    p_medium = count_params(model.medium)
    p_slow = count_params(model.slow)

    print(
        f"[params] agents={human_m(p_agents)} "
        f"fast={human_m(p_fast)} "
        f"medium={human_m(p_medium)} "
        f"slow={human_m(p_slow)} "
        f"total={human_m(p_agents+p_fast+p_medium+p_slow)}"
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    # ============================================================
    # STAGE 0 — AGENTS
    # ============================================================
    print("=== STAGE 0: TRAIN AGENTS ===")
    model.train()
    train_agents(cfg, model, train_dl, device)

    save_state_dict(os.path.join(cfg.out_dir, "agents_frozen.pt"), model.agents)
    print("[save] agents_frozen.pt")
    git_backup_checkpoints(cfg.out_dir)

    # ============================================================
    # STAGE 1 — FAST
    # ============================================================
    print("=== STAGE 1: TRAIN FAST ===")
    for p in model.agents.parameters():
        p.requires_grad = False

    model.train()
    train_fast(cfg, model, train_dl, val_dl, device)

    save_state_dict(os.path.join(cfg.out_dir, "fast_frozen.pt"), model.fast)
    print("[save] fast_frozen.pt")
    git_backup_checkpoints(cfg.out_dir)

    # ============================================================
    # STAGE 2 — MEDIUM
    # ============================================================
    print("=== STAGE 2: TRAIN MEDIUM ===")
    for p in model.fast.parameters():
        p.requires_grad = False
    for p in model.medium.parameters():
        p.requires_grad = True

    model.train()
    train_medium(cfg, model, train_dl, val_dl, device)

    save_state_dict(os.path.join(cfg.out_dir, "medium_frozen.pt"), model.medium)
    print("[save] medium_frozen.pt")
    git_backup_checkpoints(cfg.out_dir)

    # ============================================================
    # STAGE 3 — SLOW
    # ============================================================
    if cfg.slow_epochs > 0:
        print("=== STAGE 3: TRAIN SLOW ===")
        for p in model.slow.parameters():
            p.requires_grad = True

        model.train()
        train_slow(cfg, model, train_dl, val_dl, device)

        save_state_dict(os.path.join(cfg.out_dir, "slow_final.pt"), model.slow)
        print("[save] slow_final.pt")
        git_backup_checkpoints(cfg.out_dir)
    else:
        print("=== STAGE 3: SKIPPED ===")

    print(f"[done] all checkpoints saved to {cfg.out_dir}")


if __name__ == "__main__":
    main()
