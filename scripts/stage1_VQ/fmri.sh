python train_vq.py --batch-size 128 --width 512 --lr 1e-4 --total-iter 400000 --lr-scheduler 200000 --code-dim 512 --nb-code 512 --down-t 1 --depth 3 --dilation-growth-rate 3 --out-dir ./output/output_fmri --dataname fmri --vq-act relu --quantizer ema_reset_sim --exp-name VQVAE --window-size 24 --eval-iter 2000 --commit 0.01 --gpu 0

