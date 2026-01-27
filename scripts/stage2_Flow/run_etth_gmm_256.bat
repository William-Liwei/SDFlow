@echo off
REM SDFlow

set NUM_THREADS=1
set OPENBLAS_NUM_THREADS=%NUM_THREADS%

echo ================================
echo SDFlow
echo ================================
echo

set VQVAE_CKPT=.\output\output_etth\VQVAE256\net_best_ds.pth

echo start training SDFlow model...
echo

python stage2_flow/train_sdflow.py ^
    --vqvae_ckpt "%VQVAE_CKPT%" ^
    --dataname etth ^
    --output_dir .\checkpoints_etth_256 ^
    --window_size 256 ^
    --down_t 2 ^
    --quantizer ema_reset_sim ^
    --rank 1024 ^
    --d_model 1024 ^
    --n_layers 3 ^
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
    --eval_interval 20000 ^
    --eval_steps 20 ^
    --device cuda

echo.
echo ================================
echo Finished
echo ================================
echo.

echo 按任意键继续...
pause