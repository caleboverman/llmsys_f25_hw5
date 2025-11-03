import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, ylabel, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    single_mean, single_std = 48.83797280788421, 0.1449735612858846
    device0_mean, device0_std =  29.61578960418701, 9.923982936691035
    device1_mean, device1_std =  26.605942058563233, 2.322660918176044
    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
         "GPT2 Execution Time (Second)",
        'ddp_vs_rn.png')

    device_token_mean, device_token_std = 210424.76450593607, 941.8155379847459
    single_token_mean, single_token_std =  80303.4657044615, 484.94988287617525
    plot([device_token_mean, single_token_mean],
        [device_token_std, single_token_std],
        ['Data Parallel - 2GPUs', 'Single GPU'],
         "GPT2 Throughput (Tokens per Second)",
        'ddp_vs_rn2.png')

    pp_mean, pp_std = 48.38349425792694, 0.13692247867584229
    mp_mean, mp_std = 48.969767570495605, 0.10080480575561523
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
         "GPT2 Execution Time (Second)",
        'pp_vs_mp.png')

    pp_token_mean, pp_token_std = 13194.509520849286, 45.66273155822455
    mp_token_mean, mp_token_std = 13048.4779308935, 24.017911832460413
    plot([pp_token_mean, mp_token_mean],
        [pp_token_std, mp_token_std],
        ['Pipeline Parallel', 'Model Parallel'],
         "GPT2 Throughput (Tokens per Second)",
        'pp_vs_mp2.png')
