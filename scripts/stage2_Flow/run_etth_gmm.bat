
@echo off
REM SDFlow

set NUM_THREADS=1
set OPENBLAS_NUM_THREADS=%NUM_THREADS%

echo ================================
echo SDFlow
echo ================================
echo.

set VQVAE_CKPT=D:\Download\1\1\output\output_etth\VQVAE\net_best_ds.pth

echo start training SDFlow model...
echo.

python stage2_flow\train_sdflow.py ^
    --vqvae_ckpt %VQVAE_CKPT% ^
    --dataname etth ^
    --output_dir ./checkpoints_etth ^
    --window_size 24 ^
    --down_t 2 ^
    --quantizer ema_reset_sim ^
    --rank 256 ^
    --d_model 512 ^
    --n_layers 1 ^
    --num_heads 16 ^
    --dropout 0.1 ^
    --noise_std 0.01 ^
    --lambda_mean 0.1 ^
    --lambda_std 10.0 ^
    --batch_size 64 ^
    --lr 1e-4 ^
    --lr_uv 1e-3 ^
    --max_iters 100000 ^
    --print_interval 200 ^
    --eval_interval 5000 ^
    --eval_steps 20 ^
    --device cuda

echo.
echo ================================
echo Finished
echo ================================
echo.

pause