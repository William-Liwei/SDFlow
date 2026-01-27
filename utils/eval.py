from metrics.context_fid import Context_FID
from metrics.cross_correlation import CrossCorrelLoss
from metrics.metric_utils import display_scores
import numpy as np
import torch
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics2
from metrics.metric_utils import visualization
import argparse

parser = argparse.ArgumentParser(description='Process a directory.')
parser.add_argument('--dir', type=str, help='Input directory path')
args = parser.parse_args()
root_dir = args.dir

labels = np.load(root_dir+"/labels_ds.npy")
pres = np.load(root_dir+"/pres_ds.npy")
max_steps_metric = 5


context_fid_score = []

for i in range(5):
    context_fid = Context_FID(labels, pres)
    context_fid_score.append(context_fid)
    print(f'Iter {i}: ', 'context-fid =', context_fid, '\n')
display_scores(context_fid_score)

def random_choice(size, num_select=100):
    select_idx = np.random.randint(low=0, high=size, size=(num_select,))
    return select_idx

x_real = torch.from_numpy(labels)
x_fake = torch.from_numpy(pres)

correlational_score = []
size = int(x_real.shape[0] / 5)

for i in range(5):
    real_idx = random_choice(x_real.shape[0], size)
    fake_idx = random_choice(x_fake.shape[0], size)
    corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
    loss = corr.compute(x_fake[fake_idx, :, :])
    correlational_score.append(loss.item())
    print(f'Iter {i}: ', 'cross-correlation =', loss.item(), '\n')

display_scores(correlational_score)


labels = list(labels)
pres = list(pres)


discriminative_score = list()
for tt in range(max_steps_metric):  # max_steps_metric
    temp_pred = discriminative_score_metrics(labels, pres)
    msg = f"--> \t Eva. Time {tt} :, Discriminative Score. {temp_pred:.6f}"
    print(msg)
    discriminative_score.append(temp_pred)
ds_mean = np.mean(discriminative_score)
ds_std = np.std(discriminative_score)
msg = f"--> \t Eva. Iter:, Discriminative Score. {ds_mean:.6f}, std:{ds_std}"
print(msg)

predictive_score = list()
for tt in range(max_steps_metric):
    temp_pred = predictive_score_metrics2(
        labels, pres)
    msg = f"--> \t Eva. Time {tt} :, Predictive Score. {temp_pred:.6f}"
    print(msg)
    predictive_score.append(temp_pred)
ps_mean = np.mean(predictive_score)
ps_std = np.std(predictive_score)
msg = f"--> \t Eva. Iter:, Predictive Score. {ps_mean:.6f}, std:{ps_std}"
print(msg)