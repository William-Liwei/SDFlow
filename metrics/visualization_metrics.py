from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
font_size = 22
plt.rc('font', size=font_size) 
plt.rc('axes', titlesize=font_size) 
plt.rc('axes', labelsize=font_size) 
plt.rc('xtick', labelsize=font_size) 
plt.rc('ytick', labelsize=font_size) 
plt.rc('legend', fontsize=font_size) 
plt.rc('figure', titlesize=font_size)

def visualization(ori_data, generated_data1, analysis, args):
    
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    ori_data = ori_data[:,:,:-1]
    generated_data1 = np.asarray(generated_data1)

    ori_data = ori_data[idx]
    generated_data1 = generated_data1[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data1[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            prep_data_hat = np.concatenate(
                (prep_data_hat, np.reshape(np.mean(generated_data1[i, :, :], 1), [1, seq_len]))
            )

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == "tsne":

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            tsne_results[anal_sample_no:, 0],
            tsne_results[anal_sample_no:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="GT-GAN",
        )
        plt.legend(prop={'size': 22},markerscale=2)
        plt.title("t-SNE plot")
        plt.rcParams['pdf.fonttype'] = 42
        plt.savefig(str(args.save_dir)+"/"+args.model1+"_tsne.png", dpi=100,bbox_inches='tight')
        plt.close()

    elif analysis == "histogram":
        f, ax = plt.subplots(1)
        sns.distplot(prep_data, hist = False, kde = True,kde_kws = {'linewidth': 6},label = 'Original')
        sns.distplot(prep_data_hat, hist = False, kde = True,kde_kws = {'linewidth': 6,'linestyle':'--'},label = 'GT-GAN')
        # Plot formatting
        plt.legend(prop={'size': 22})
        plt.xlabel('Data Value')
        plt.ylabel('Data Density Estimate')
        plt.rcParams['pdf.fonttype'] = 42
        plt.savefig(str(args.save_dir)+"/"+args.model1+"_histo.png", dpi=100,bbox_inches='tight')
        plt.close()