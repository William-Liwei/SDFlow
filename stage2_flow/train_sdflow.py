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
from models.sdflow import SDFlowModel, KDEPrior


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

    

def train():
    parser = argparse.ArgumentParser(description="SDFlow")
    parser.add_argument('--vqvae_ckpt', type=str, required=True)
    parser.add_argument('--dataname', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints_SDFlow')
    
    parser.add_argument('--quantizer', type=str, default='ema_reset_sim')
    parser.add_argument('--window_size', type=int, default=24)
    parser.add_argument('--down_t', type=int, default=2)
    
    parser.add_argument('--rank', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--lambda_mean', type=float, default=0.1)
    parser.add_argument('--lambda_std', type=float, default=0.1)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_uv', type=float, default=5e-4)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--print_interval', type=int, default=200)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--eval_steps', type=int, default=50)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = utils_model.get_logger(args.output_dir)
    writer = SummaryWriter(args.output_dir)
    
    # Load data
    codebook, train_indices, seq_len = auto_encode_and_cache(
        vqvae_ckpt=args.vqvae_ckpt,
        dataname=args.dataname,
        cache_dir=args.output_dir,
        device=device,
        window_size=args.window_size,
        down_t=args.down_t,
        quantizer=args.quantizer
    )
    
    num_codes, code_dim = codebook.shape
    n_samples = len(train_indices)
    
    print(f"\nData Statistics:")
    print(f"  samples: {n_samples}")
    print(f"  Codebook: {num_codes} codes × {code_dim} dim")
    
    codebook = codebook.to(device)
    train_indices = train_indices.to(device)
    
    # Load VQ-VAE
    print(f"\nLoading VQ-VAE...")
    sys_argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0], '--dataname', args.dataname]
    vqvae_args = option_trans.get_args_parser()
    sys.argv = sys_argv_backup
    
    if args.quantizer:
        vqvae_args.quantizer = args.quantizer
    if args.window_size:
        vqvae_args.window_size = args.window_size
    if args.down_t:
        vqvae_args.down_t = args.down_t
    
    vqvae_checkpoint = torch.load(args.vqvae_ckpt, map_location='cpu', weights_only=False)
    vq_model = vqvae.VQVAE(vqvae_args, num_codes, code_dim,
                           vqvae_args.down_t, vqvae_args.stride_t,
                           vqvae_args.width, vqvae_args.depth,
                           vqvae_args.dilation_growth_rate).to(device)
    vq_model.load_state_dict(vqvae_checkpoint['net'])
    vq_model.eval()
    
    # Create model
    model = SDFlowModel(
        num_samples=n_samples,
        num_codes=num_codes,
        code_dim=code_dim,
        seq_len=seq_len,
        rank=args.rank,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        noise_std=args.noise_std,
        lambda_mean=args.lambda_mean,
        lambda_std=args.lambda_std,
        t_scheduler='cosine'
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if n not in ['U', 'V']],
         'lr': args.lr},
        {'params': [model.U, model.V], 'lr': args.lr_uv}
    ], weight_decay=0.01)
    
    warmup_steps = 500
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (args.max_iters - step) / (args.max_iters - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    
    model.train()
    best_ds = float('inf')
    
    for iter_idx in range(args.max_iters):
        # Sample batch
        batch_sample_ids = torch.randint(0, n_samples, (args.batch_size,), device=device)
        batch_indices = train_indices[batch_sample_ids]
        
        # Compute loss
        loss, metrics = model.compute_loss(codebook, batch_indices, batch_sample_ids)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if (iter_idx + 1) % args.print_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            lr_uv = optimizer.param_groups[1]['lr']
            msg = f"Iter {iter_idx+1}: Loss={metrics['loss']:.6f}, CE={metrics['ce_loss']:.6f}, Mean={metrics['mean_loss']:.6f}, Std={metrics['std_loss']:.6f}, structure={metrics['structure_loss']:.6f}, Acc={metrics['accuracy']:.6f}"
            print(msg)
            logger.info(msg)
            writer.add_scalar('Train/Loss', metrics['loss'], iter_idx + 1)
            writer.add_scalar('Train/CE_Loss', metrics['ce_loss'], iter_idx + 1)
            writer.add_scalar('Train/Mean_Loss', metrics['mean_loss'], iter_idx + 1)
            writer.add_scalar('Train/Std_Loss', metrics['std_loss'], iter_idx + 1)
            writer.add_scalar('Train/Accuracy', metrics['accuracy'], iter_idx + 1)
        
        # Evaluation
        if (iter_idx + 1) % args.eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation - Iter {iter_idx+1}")
            print(f"{'='*60}")
            
            model.eval()
            
            # 0.001 is the dimensionless calibration factor alpha, not the
            # final noise magnitude. The realized bandwidth is
            # h = alpha * mean nearest-neighbor distance in U, and h is the
            # per-coordinate standard deviation. Noise is added independently
            # in all rank dimensions, so the expected radial perturbation is
            # approximately h * sqrt(rank); therefore 0.001 alone should not
            # be interpreted as universally small.
            kde = KDEPrior(model.U, device=device, bandwidth_factor=0.001)
            
            # Check U statistics
            with torch.no_grad():
                u_mean = model.U.mean(dim=0).abs().mean().item()
                u_std = model.U.std(dim=0).mean().item()
                print(f"\nU Statistics:")
                print(f"  Mean: {u_mean:.6f} (target: 0)")
                print(f"  Std: {u_std:.6f} (target: 1)")
            

            num_gen = min(50000, n_samples)

            real_loader = dataset_VQ.DATALoader(
                args.dataname,
                batch_size=128,
                num_workers=0,
                window_size=args.window_size,
                unit_length=2**args.down_t,
                dataset_type='train'
            )

            real_timeseries = []
            for batch in real_loader:
                batch_np = batch.numpy()
                for i in range(batch_np.shape[0]):
                    real_timeseries.append(batch_np[i])
            real_timeseries = np.array(real_timeseries)

            num_gen = len(real_timeseries)

            gen_batch_size = 128
            gen_indices_list = []

            with torch.no_grad():
                for i in tqdm(range(0, num_gen, gen_batch_size), desc="Generating"):
                    bs = min(gen_batch_size, num_gen - i)
                    _, gen_idx = model.sample_kde(
                        codebook=codebook,
                        batch_size=bs,
                        seq_len=seq_len,
                        steps=args.eval_steps,
                        temperature=0.9,
                        device=device,
                        kde_solver=kde 
                    )
                    gen_indices_list.append(gen_idx.cpu())

            gen_indices = torch.cat(gen_indices_list, dim=0).to(device)
            gen_timeseries = decode_in_batches(vq_model, gen_indices, batch_size=64)

            # DS
            print("Computing DS...")
            ds_scores = []
            for _ in range(5): 
                ds = discriminative_score_metrics(real_timeseries, gen_timeseries)
                ds_scores.append(ds)
            ds_mean = np.mean(ds_scores)
            ds_std = np.std(ds_scores)

            print(f"  DS: {ds_mean:.6f} ± {ds_std:.6f}")
            logger.info(f"  DS: {ds_mean:.6f} ± {ds_std:.6f}")
            writer.add_scalar('Eval/DS', ds_mean, iter_idx + 1)
            writer.add_scalar('Eval/DSstd', ds_std, iter_idx + 1)

            # Predictive Score
            print("\nComputing Predictive Score...")
            try:
                # PS expects input format as [number of samples], each sample is (seq_len, dim)
                real_list = [real_timeseries[i].T for i in range(len(real_timeseries))]
                gen_list = [gen_timeseries[i].T for i in range(len(gen_timeseries))]
                pred = []
                for _ in range(5):
                    # pred_score = predictive_score_metrics(real_list, gen_list)
                    pred_score = predictive_score_metrics(real_timeseries, gen_timeseries)
                    pred.append(pred_score)
                    msg = f"PS: {pred_score:.6f}"
                    print(msg)

                pred_mean = np.mean(pred)
                pred_std = np.std(pred)

                print(f"  PS: {pred_mean:.6f} ± {pred_std:.6f}")
                logger.info(f"  PS: {pred_mean:.6f} ± {pred_std:.6f}")
                writer.add_scalar('Eval/PS', pred_mean, iter_idx + 1)
                writer.add_scalar('Eval/PSstd', pred_std, iter_idx + 1)
            except Exception as e:
                print(f"Pred failed: {e}")
                pred_score = None

            # Context FID
            print("\nComputing Context FID...")
            try:
                real_fid = np.transpose(real_timeseries, (0, 2, 1))
                gen_fid = np.transpose(gen_timeseries, (0, 2, 1))
                fid = []
                for _ in range(5):
                    fid_score = Context_FID(real_fid, gen_fid)
                    fid.append(fid_score)
                    msg = f"FID: {fid_score:.6f}"
                    print(msg)

                fid_mean = np.mean(fid)
                fid_std = np.std(fid)

                print(f"  FID: {fid_mean:.6f} ± {fid_std:.6f}")
                logger.info(f"  FID: {fid_mean:.6f} ± {fid_std:.6f}")
                writer.add_scalar('Eval/FID', fid_mean, iter_idx + 1)
                writer.add_scalar('Eval/FIDstd', fid_std, iter_idx + 1)
            except Exception as e:
                print(f"FID failed: {e}")
                fid_score = None
            
            # MSE
            mse = np.mean((real_timeseries - gen_timeseries) ** 2)
            print(f"  MSE: {mse:.6f}")
            writer.add_scalar('Eval/MSE', mse, iter_idx + 1)
            
            if ds_mean < best_ds:
                best_ds = ds_mean
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'iter': iter_idx + 1,
                    'best_ds': best_ds,
                    'args': vars(args)
                }, os.path.join(args.output_dir, 'best_ds.pth'))
                print(f"✓ New Best DS: {best_ds:.6f}")
            
            print(f"{'='*60}\n")
            model.train()
    
    print(f"\nBest DS: {best_ds:.6f}")
    writer.close()


if __name__ == "__main__":
    train()
