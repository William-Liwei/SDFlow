"""
SDFlow
"""

import math
import os
import sys
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import hashlib
from sklearn.mixture import GaussianMixture

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
from dataset import dataset_VQ
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.context_fid import Context_FID
from metrics.predictive_metrics import predictive_score_metrics2 as predictive_score_metrics


class KDEPrior:
    def __init__(self, u_samples, device='cuda', bandwidth_factor=1.0):
        """
        u_samples: [N, D]
        bandwidth_factor
        """
        self.u_samples = u_samples.detach().float().to(device)
        self.num_samples = u_samples.shape[0]
        self.dim = u_samples.shape[1]
        self.device = device

        n_subset = min(2000, self.num_samples)
        subset_idx = torch.randperm(self.num_samples)[:n_subset]
        subset = self.u_samples[subset_idx]
        
        dists = torch.cdist(subset, subset)
        dists.fill_diagonal_(float('inf'))
        
        min_dists, _ = dists.min(dim=1)
        avg_nn_dist = min_dists.mean().item()
        
        self.bandwidth = avg_nn_dist * bandwidth_factor
        print(f"  > Optimal Bandwidth (sigma): {self.bandwidth:.6f}")

    def sample(self, n_samples):
        """
        KDE
        """
        indices = torch.randint(0, self.num_samples, (n_samples,), device=self.device)
        centers = self.u_samples[indices]
        noise = torch.randn(n_samples, self.dim, device=self.device)
        samples = centers + noise * self.bandwidth
        
        return samples


def auto_encode_and_cache(vqvae_ckpt, dataname, cache_dir, device='cuda',
                          window_size=None, down_t=None, quantizer=None):
    print("\n" + "="*60)
    print("Preprocessing")
    print("="*60)

    vqvae_checkpoint = torch.load(vqvae_ckpt, map_location='cpu', weights_only=False)
    codebook_raw = vqvae_checkpoint['net']['vqvae.quantizer.codebook']
    num_codes, code_dim = codebook_raw.shape

    sys_argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0], '--dataname', dataname]
    vqvae_args = option_trans.get_args_parser()
    sys.argv = sys_argv_backup

    if quantizer:
        vqvae_args.quantizer = quantizer
    if window_size:
        vqvae_args.window_size = window_size
    if down_t:
        vqvae_args.down_t = down_t

    window_size = vqvae_args.window_size
    down_t = vqvae_args.down_t

    quantizer_str = quantizer if quantizer else "default"
    config_str = f"{dataname}_{window_size}_{down_t}_{num_codes}_{code_dim}_{quantizer_str}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    cache_subdir = os.path.join(cache_dir, f'auto_cache_{dataname}_{config_hash}')
    codebook_path = os.path.join(cache_subdir, 'codebook.pth')
    indices_path = os.path.join(cache_subdir, 'train_indices.pth')
    config_path = os.path.join(cache_subdir, 'config.json')

    cache_valid = False
    if os.path.exists(config_path) and os.path.exists(codebook_path) and os.path.exists(indices_path):
        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            if (saved_config['dataname'] == dataname and
                saved_config['window_size'] == window_size and
                saved_config['down_t'] == down_t):
                cache_valid = True
        except:
            pass

    if cache_valid:
        codebook = torch.load(codebook_path, weights_only=False)
        train_indices = torch.load(indices_path, weights_only=False)
        print(f"  Codebook: {codebook.shape}")
        print(f"  Indices: {train_indices.shape}")
        print("="*60 + "\n")
        return codebook, train_indices, train_indices.shape[1]

    print("\nEncoding data...")
    os.makedirs(cache_subdir, exist_ok=True)

    vq_model = vqvae.VQVAE(
        vqvae_args, num_codes, code_dim,
        vqvae_args.down_t, vqvae_args.stride_t,
        vqvae_args.width, vqvae_args.depth,
        vqvae_args.dilation_growth_rate
    ).to(device)
    vq_model.load_state_dict(vqvae_checkpoint['net'])
    vq_model.eval()

    train_loader = dataset_VQ.DATALoader(
        dataname, batch_size=128, num_workers=0,
        window_size=window_size, unit_length=2**down_t,
        dataset_type='train'
    )

    train_indices_list = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Encoding"):
            batch_data = batch.to(device).float()
            indices = vq_model.encode(batch_data)
            train_indices_list.append(indices.cpu())

    train_indices = torch.cat(train_indices_list, dim=0)
    codebook = codebook_raw.cpu()

    torch.save(codebook, codebook_path)
    torch.save(train_indices, indices_path)
    config = {
        'dataname': dataname, 'window_size': window_size,
        'down_t': down_t, 'num_codes': num_codes,
        'code_dim': code_dim, 'vqvae_ckpt': vqvae_ckpt,
        'seq_len': train_indices.shape[1],
        'n_samples': train_indices.shape[0]
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("="*60 + "\n")
    return codebook, train_indices, train_indices.shape[1]


def decode_in_batches(vq_model, indices, batch_size=32):
    decoded_list = []
    num_samples = len(indices)
    device = indices.device

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_recon = vq_model.forward_decoder(batch_indices)
            decoded_list.append(batch_recon.detach().cpu())

    return torch.cat(decoded_list, dim=0).numpy()


class AdaLayerNorm(nn.Module):
    def __init__(self, d_model, time_emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, d_model * 2)
        )

    def forward(self, x, t_emb):
        scale, shift = self.mlp(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = AdaLayerNorm(d_model, time_emb_dim)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = AdaLayerNorm(d_model, time_emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, t_emb):
        normed_x = self.norm1(x, t_emb)
        x = x + self.attn(normed_x, normed_x, normed_x)[0]
        x = x + self.ffn(self.norm2(x, t_emb))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SDFlowModel(nn.Module):
    
    def __init__(
        self,
        num_samples,
        num_codes,
        code_dim,
        seq_len,
        rank=128,
        d_model=512,
        n_layers=6,
        num_heads=8,
        dropout=0.1,
        noise_std=0.01,
        lambda_mean=0.1,
        lambda_std=0.1,
        t_scheduler='cosine'
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.seq_len = seq_len
        self.rank = rank
        self.d_model = d_model
        self.noise_std = noise_std
        self.lambda_mean = lambda_mean
        self.lambda_std = lambda_std
        self.t_scheduler = t_scheduler

        # Low-rank factorization
        init_U = torch.randn(num_samples, rank) * 0.01
        self.U = nn.Parameter(init_U)
        
        init_V = torch.randn(rank, seq_len * code_dim) * 0.01
        self.V = nn.Parameter(init_V)
        
        # Global token
        init_val = torch.randn(1, 1, code_dim)
        init_val = F.normalize(init_val, dim=-1)
        self.global_token = nn.Parameter(init_val)

        # Time embedding
        time_emb_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Input projection
        self.input_proj = nn.Linear(code_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, num_heads, d_model * 4, time_emb_dim, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_codes)

    def get_z0(self, sample_ids, training=False):
        U_batch = self.U[sample_ids]
        if training and self.noise_std > 0:
            noise = torch.randn_like(U_batch) * self.noise_std
            U_batch = U_batch + noise
        
        z0_flat = torch.matmul(U_batch, self.V)
        z0 = z0_flat.reshape(-1, self.seq_len, self.code_dim)
        z0 = F.normalize(z0, dim=-1)
        return z0

    def sample_z0_unconditional(self, batch_size, device):
        u = torch.randn(batch_size, self.rank, device=device)
        z0_flat = torch.matmul(u, self.V)
        z0 = z0_flat.reshape(batch_size, self.seq_len, self.code_dim)
        z0 = F.normalize(z0, dim=-1)
        return z0

    def forward(self, z_t, t):
        """Forward pass"""
        B, L, D = z_t.shape

        global_token = self.global_token.expand(B, -1, -1)
        z_t_with_cls = torch.cat([global_token, z_t], dim=1)

        if t.dim() > 1:
            t = t.squeeze()
        t_emb = self.time_mlp(t.unsqueeze(-1))

        x = self.input_proj(z_t_with_cls)
        x = self.pos_encoder(x)

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.final_norm(x)
        logits_with_cls = self.output_proj(x)
        logits = logits_with_cls[:, 1:, :]

        return logits

    
    def compute_loss(self, codebook, indices, sample_ids, mixup_prob=0.5):
        """
        Training loss with Manifold Mixup
        """
        B, L = indices.shape
        device = indices.device

        z1 = codebook[indices] # [B, L, D]
        z0 = self.get_z0(sample_ids, training=True) # [B, L, D]


        u_batch = self.U[sample_ids]
        z1_flat = z1.view(z1.shape[0], -1)

        real_dist = torch.cdist(z1_flat, z1_flat, p=2)
        real_dist = real_dist / (real_dist.mean() + 1e-6)

        u_dist = torch.cdist(u_batch, u_batch, p=2)
        u_dist = u_dist / (u_dist.mean() + 1e-6)

        structure_loss = F.mse_loss(u_dist, real_dist)


        if self.training and torch.rand(1).item() < mixup_prob:
            perm = torch.randperm(B, device=device)
            
            # lambda ~ Beta(alpha, alpha)
            lam = np.random.beta(1.0, 1.0)
            lam = max(lam, 1 - lam)
            u_current = self.U[sample_ids]
            u_perm = self.U[sample_ids[perm]]
            u_mix = lam * u_current + (1 - lam) * u_perm

            z0_flat = torch.matmul(u_mix, self.V)
            z0_mix = z0_flat.reshape(B, L, -1)
            z0_mix = F.normalize(z0_mix, dim=-1)
            

            z1_perm = z1[perm]
            z1_mix = lam * z1 + (1 - lam) * z1_perm

            z0 = z0_mix
            z1 = z1_mix

            current_indices = indices
        else:
            current_indices = indices

        # Flow Matching
        t = torch.rand([B, 1, 1], device=device, dtype=z1.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos((t ** 2.0) * 0.5 * torch.pi)

        z_t = (1 - t) * z0 + t * z1
        
        logits = self.forward(z_t, t.squeeze(-1)) # [B, L, num_codes]

        loss_a = F.cross_entropy(logits.reshape(-1, self.num_codes), indices.reshape(-1), reduction='none')
        
        if 'perm' in locals():
            loss_b = F.cross_entropy(logits.reshape(-1, self.num_codes), indices[perm].reshape(-1), reduction='none')
            ce_loss = (lam * loss_a + (1 - lam) * loss_b).mean()
        else:
            ce_loss = loss_a.mean()

        # Distribution Regularization
        mean_loss = self.U.mean(dim=0).abs().mean()
        std_loss = (self.U.std(dim=0) - 1.0).abs().mean()

        # Total loss
        loss = ce_loss + self.lambda_mean * mean_loss + self.lambda_std * std_loss + 10 * structure_loss

        # Accuracy
        pred_indices = logits.argmax(dim=-1)
        accuracy = (pred_indices == current_indices).float().mean()

        return loss, {
            'loss': loss.item(),
            'ce_loss': ce_loss.item(),
            'mean_loss': mean_loss.item(),
            'std_loss': std_loss.item(),
            'structure_loss' : structure_loss.item(),
            'accuracy': accuracy.item()
        }
    
    @torch.no_grad()
    def sample_kde(self, codebook, batch_size, seq_len, steps=50, temperature=0.9, device='cuda', kde_solver=None):
        """
        KDE-based Sampling
        kde_solver
        """
        
        if kde_solver is not None:
            u_sample = kde_solver.sample(batch_size)

            z0_flat = torch.matmul(u_sample, self.V)
            z = z0_flat.reshape(batch_size, self.seq_len, self.code_dim)
            z = F.normalize(z, dim=-1)
        else:
            z = self.sample_z0_unconditional(batch_size, device)

        # Flow Matching ODE
        t_span = torch.linspace(0, 1, steps + 1, device=device)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos((t_span ** 2.0) * 0.5 * torch.pi)

        for i in range(len(t_span) - 1):
            t_curr = t_span[i]
            dt = t_span[i + 1] - t_curr
            t_batch = t_curr.unsqueeze(0).repeat(batch_size)
            
            logits = self.forward(z, t_batch)
            probs = F.softmax(logits / temperature, dim=-1)
            mu_t = torch.matmul(probs, codebook)

            if t_curr < 0.999:
                velocity = (mu_t - z) / (1 - t_curr + 1e-5)
            else:
                velocity = torch.zeros_like(z)
            z = z + velocity * dt

        z = F.normalize(z, dim=-1)
        z_flat = z.reshape(-1, self.code_dim)
        distances = torch.cdist(z_flat, codebook)
        indices = distances.argmin(dim=-1).reshape(batch_size, seq_len)

        return z, indices
    
    @torch.no_grad()
    def sample(self, codebook, batch_size, seq_len, steps=50, temperature=0.9,
               device='cuda'):
        """
        Sampling Strategy: Latent Mixup (Manifold Interpolation)
        """

        idx_a = torch.randint(0, self.num_samples, (batch_size,), device=device)
        idx_b = torch.randint(0, self.num_samples, (batch_size,), device=device)
        

        u_a = self.U[idx_a] # [B, rank]
        u_b = self.U[idx_b] # [B, rank]
        
        alpha = torch.rand(batch_size, 1, device=device) * 1.0 

        u_mix = alpha * u_a + (1 - alpha) * u_b

        noise = torch.randn_like(u_mix) * 0.01 
        u_new = u_mix + noise

        z0_flat = torch.matmul(u_new, self.V)
        z = z0_flat.reshape(batch_size, self.seq_len, self.code_dim)
        z = F.normalize(z, dim=-1)


        t_span = torch.linspace(0, 1, steps + 1, device=device)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos((t_span ** 2.0) * 0.5 * torch.pi)

        for i in range(len(t_span) - 1):
            t_curr = t_span[i]
            dt = t_span[i + 1] - t_curr
            t_batch = t_curr.unsqueeze(0).repeat(batch_size)
            
            logits = self.forward(z, t_batch)
            probs = F.softmax(logits / temperature, dim=-1)
            mu_t = torch.matmul(probs, codebook)

            if t_curr < 0.999:
                velocity = (mu_t - z) / (1 - t_curr + 1e-5)
            else:
                velocity = torch.zeros_like(z)
            z = z + velocity * dt

        z = F.normalize(z, dim=-1)
        z_flat = z.reshape(-1, self.code_dim)
        distances = torch.cdist(z_flat, codebook)
        indices = distances.argmin(dim=-1).reshape(batch_size, seq_len)

        return z, indices