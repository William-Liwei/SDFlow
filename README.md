# SDFlow: Similarity-Driven Flow Matching for Time Series Generation

**Wei Li**$^{1,2,*}$, **Shibo Feng**$^{3,*}$, **Pengcheng Wu**$^{3}$, **Xingyu Gao**$^{4}$, **Min Wu**$^{5}$, **Peilin Zhao**$^{1,\dagger}$

$^{1}$Shanghai Jiao Tong University; $^{2}$Shanghai University; $^{3}$Nanyang Technological University; $^{4}$Chinese Academy of Sciences, Beijing; $^{5}$Institute for Infocomm Research, A*STAR, Singapore.

$^{*}$Equal contribution.  
$^{\dagger}$Corresponding author.

To address exposure bias and high-dimensional representations in VQ-based discrete time-series generation, we propose SDFlow, a non-autoregressive flow-matching framework that explores low-rank manifold anchoring and discrete supervision. SDFlow enables efficient long-sequence generation and substantial inference acceleration while maintaining high fidelity. It achieves over 100x speedup compared with diffusion models, delivers more than an order-of-magnitude improvement in distribution quality over existing flow-based baselines on long-sequence tasks, and provides a 95% distribution-quality improvement and 3--10x speedup over the previous SDformer-ar model. SDFlow can support applications such as energy dispatch, financial simulation, and traffic monitoring, providing an efficient and reliable generative foundation for long-horizon trend forecasting and high-fidelity data augmentation.

Run the following commands from the repository root after preparing the dataset under `dataset/` and installing the environment in `environment.yaml`.

## Stage 1: Train the VQ-VAE tokenizer

```bash
python -m stage1_vq.train_vq \
  --batch-size 128 --width 512 --lr 1e-4 --total-iter 100000 \
  --lr-scheduler 200000 --code-dim 512 --nb-code 512 --down-t 2 \
  --depth 3 --dilation-growth-rate 3 --out-dir ./output/output_energy \
  --dataname energy --vq-act relu --quantizer ema_reset_sim \
  --exp-name VQVAE --window-size 24 --commit 0.001 --gpu 0
```

## Stage 2: Train SDFlow

```bash
python -m stage2_flow.train_sdflow \
  --vqvae_ckpt ./output/output_energy/VQVAE/net_best_ds.pth \
  --dataname energy --output_dir ./checkpoints_energy --window_size 24 \
  --down_t 2 --quantizer ema_reset_sim --rank 256 --d_model 512 \
  --n_layers 1 --num_heads 16 --dropout 0.1 --noise_std 0.01 \
  --lambda_mean 0.1 --lambda_std 10.0 --batch_size 64 --lr 1e-4 \
  --lr_uv 1e-3 --max_iters 100000 --print_interval 200 \
  --eval_interval 5000 --eval_steps 20 --device cuda --seed 42
```

## Citation

```bibtex
@inproceedings{li2026sdflow,
  title     = {SDFlow: Similarity-Driven Flow Matching for Time Series Generation},
  author    = {Li, Wei and Feng, Shibo and Wu, Pengcheng and Gao, Xingyu and Wu, Min and Zhao, Peilin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
