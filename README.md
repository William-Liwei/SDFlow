# SDFlow: Similarity-Driven Flow Matching for Time Series Generation

## 1.Install the environment using yaml file

```
conda env create -f environment.yaml
```

## 2.Train Stage 1

```
python train_vq.py --batch-size 128 --width 512 --lr 1e-4 --total-iter 100000 --lr-scheduler 200000 --code-dim 512 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir ./output/output_energy --dataname energy --vq-act relu --quantizer ema_reset_sim --exp-name VQVAE --window-size 24 --commit 0.001 --gpu 0 
```

## 3.Train Stage 2

```
python stage2_flow/train_sdflow.py --vqvae_ckpt %VQVAE_CKPT% --dataname energy --output_dir ./checkpoints_energy --window_size 24 --down_t 2 --quantizer ema_reset_sim --rank 256 --d_model 512 --n_layers 1 --num_heads 16 --dropout 0.1 --noise_std 0.01 --lambda_mean 0.1 --lambda_std 10.0 --batch_size 64 --lr 1e-4 --lr_uv 1e-3 --max_iters 100000 --print_interval 200 --eval_interval 5000 --eval_steps 20 --device cuda
```
