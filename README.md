# SDFlow: Similarity-Driven Flow Matching for Time Series Generation

<p align="center">
  <strong>Wei Li</strong><sup>1,2,*</sup>, <strong>Shibo Feng</strong><sup>3,*</sup>, <strong>Pengcheng Wu</strong><sup>3</sup>, <strong>Xingyu Gao</strong><sup>4</sup>, <strong>Min Wu</strong><sup>5</sup>, <strong>Peilin Zhao</strong><sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>Shanghai Jiao Tong University; <sup>2</sup>Shanghai University; <sup>3</sup>Nanyang Technological University;<br>
  <sup>4</sup>Chinese Academy of Sciences, Beijing; <sup>5</sup>Institute for Infocomm Research, A*STAR, Singapore.
</p>

<p align="center"><sup>*</sup>Equal contribution. &nbsp; <sup>†</sup>Corresponding author.</p>

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
