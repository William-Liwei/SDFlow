import os

import clip
import numpy as np
import torch

import math
from tqdm import tqdm
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics2
import tensorflow as tf
from metrics.context_fid import Context_FID
from metrics.cross_correlation import CrossCorrelLoss
from metrics.metric_utils import display_scores


def calculate_mse_mae(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)

    mae = np.mean(np.abs(y_true - y_pred))

    return mse, mae

@torch.no_grad()
def evaluation_vqvae(args,out_dir, val_loader, net, logger, writer, nb_iter, best_iter, best_ds, best_mse, save = True, isTest = False) :
    net.eval()
    labels = []
    pres = []
    nb_used = set({})
    for batch in val_loader:
        batch = batch.cuda().float()
        a = net.encode(batch)
        pre = net.forward_decoder(a)
        label = batch.detach().cpu().numpy()
        pre = pre.detach().cpu().numpy()
        labels.append(label)
        pres.append(pre)
        nb_used = nb_used.union(set(a.cpu().numpy().reshape(-1)))

    labels = np.concatenate(labels,0)
    pres = np.concatenate(pres,0)


    mse,mae = calculate_mse_mae(pres,labels)
    msg = f"--> \t Eva. Iter {nb_iter} :, MAE. {mae:.4f}, MSE. {mse:.4f},used_code. {len(nb_used)}"
    logger.info(msg)


    labels = list(labels)
    pres = list(pres)


    discriminative_score = list()
    for tt in range(3):
        temp_pred = discriminative_score_metrics(labels, pres)
        discriminative_score.append(temp_pred)
    ds_mean = np.mean(discriminative_score)
    ds_std = np.std(discriminative_score)
    msg = f"--> \t Eva. Iter {nb_iter} :, Discriminative Score. {ds_mean:.6f}, std:{ds_std}"
    logger.info(msg)

    ds = ds_mean

    if ds < best_ds :
        msg = f"--> --> \t Discriminative Score Improved from {best_ds:.5f} to {ds:.5f} !!!"
        logger.info(msg)
        best_ds, best_iter = ds, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_ds.pth'))

    if mse < best_mse :
        msg = f"--> --> \t MSE Improved from {best_mse:.5f} to {mse:.5f} !!!"
        logger.info(msg)
        best_mse = mse
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_mse.pth'))

    net.train()
    return best_iter, best_ds, best_mse, writer, logger

@torch.no_grad()
def evaluation_transformer(args,out_dir, val_loader, trans, net, logger, writer, nb_iter, best_iter, best_ds,
                     save=True):
    net.eval()
    trans.eval()

    labels = []
    pres = []
    for i,batch in enumerate(val_loader):
        batch = batch.cuda().float()
        label = batch

        input_idx = args.nb_code * torch.ones(label.shape[0], int(label.shape[1]/(2**args.down_t))).cuda().int()
        pre_idx = trans.sample(input_idx,True)
        pre = net.forward_decoder(pre_idx)

        label = label.detach().cpu().numpy()
        pre = pre.detach().cpu().numpy()
        labels.append(label)
        pres.append(pre)
    labels = np.concatenate(labels, 0)
    pres = np.concatenate(pres, 0)

    labels = list(labels)
    pres = list(pres)

    if args.if_test:
        np.save(os.path.join(out_dir, 'labels_ds.npy'), np.stack(labels, axis=0))
        np.save(os.path.join(out_dir, 'pres_ds.npy'), np.stack(pres, axis=0))
        print("Generation Completed!")
        exit()

    discriminative_score = list()
    for tt in range(5):#max_steps_metric
        temp_pred = discriminative_score_metrics(labels, pres)
        discriminative_score.append(temp_pred)
    ds_mean = np.mean(discriminative_score)
    ds_std = np.std(discriminative_score)
    msg = f"--> \t Eva. Iter {nb_iter} :, Discriminative Score. {ds_mean:.6f}, std:{ds_std}"
    logger.info(msg)


    ds = ds_mean

    if ds < best_ds :
        msg = f"--> --> \t Discriminative Score Improved from {best_ds:.5f} to {ds:.5f} !!!"
        logger.info(msg)
        best_ds, best_iter = ds, nb_iter
        if save:
            torch.save({'trans': trans.state_dict()}, os.path.join(out_dir, 'net_best_ds.pth'))
            np.save(os.path.join(out_dir, 'labels_ds.npy'), np.stack(labels, axis=0))
            np.save(os.path.join(out_dir, 'pres_ds.npy'), np.stack(pres, axis=0))

    trans.train()
    return best_iter, best_ds, writer, logger

