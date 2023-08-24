### --------------------------------------------------------------------------
#%% Figures.
### --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from constants import FP_FIG, FP_DATA_OUT
from utils import start
from cluster_robust_mean_reciprocal_rank import cluster_robust_mean_reciprocal_rank_penalty
#%%
def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif',font_scale=1.2)

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
#%%
def plot_kg_metrics(df_graph_mets):
    # Get data.
    df_graph_mets = pd.read_csv(FP_DATA_OUT + 't_Graph_metrics.csv')
    print(df_graph_mets.to_latex(escape=False))
    set_style()
    todays_date = start()
    df = df_graph_mets.transpose().reset_index()
    headers = df.iloc[0].values
    df.columns = headers
    df.drop(index=0, axis=0, inplace=True)
    value_vars = [
                    'Node count',
                    'Relation count',
                    'Mean degree',
                    'Transitivity',
                    'Mean clustering coefficent'
    ]
    titles = [
                    'Node count',
                    'Relation count',
                    'Deg',
                    'Trans',
                    'CCO'
    ]
    df_melt = df.melt(id_vars='0', value_vars=value_vars)
    order_list = ['web-google', 'web-amazon', 'sx-stackoverflow', 'web-facebook', 'Industry_KG', 'BTC_alpha', 'BTC_otc']
    g = sns.FacetGrid(df_melt, col="variable", sharex=False)
    g.map(sns.barplot, 'value', '0', color='#25586E', order = order_list)
    g.add_legend()
    g.set_titles(col_template="")
    g.set_xlabels('')
    g.set_ylabels('')
    axes = g.axes.flatten()
    for idx, name in enumerate(titles):
        axes[idx].set_title(str(name), fontsize=16)
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=30, fontsize=14)
    xlabels = ['{:.0f}'.format(x) + ' m' for x in axes[0].get_xticks() / 1000000]
    axes[0].set_xticklabels(xlabels, fontsize=14)
    axes[0].set(xlim=(0, 6000000))
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=16)
    xlabels = ['{:.0f}'.format(x) + ' k' for x in axes[1].get_xticks() / 1000]
    axes[1].set_xticklabels(xlabels, fontsize=12)
    axes[1].set(xlim=(0, 450000))
    plt.savefig(FP_FIG + 'FIG_' + str(todays_date) + '_benchmark_KG_metrics.svg')
    plt.savefig(FP_FIG + 'FIG_' + str(todays_date) + '_benchmark_KG_metrics.png')
    plt.show(block=True)
    plt.interactive(False)

#%% Plot Rank-based Metrics, HitsatK and CRMRR.
def plot_rank_based_metrics():
    df_rb_mets_updated = pd.read_csv(FP_DATA_OUT + 't_rb_mets.csv')
    print(df_rb_mets_updated.to_latex(escape=False))
    df = df_rb_mets_updated
    value_vars = [
        'implausible true completion candidates',
        'mr',
        'mrr',
        'hk',
        'crmrr'
    ]
    df_melt = df.melt(id_vars=['0', 'model'], value_vars=value_vars)
    # Plot rank-based metrics, Hk and CRMRR.
    set_style()
    todays_date = start()

    g = sns.FacetGrid(df_melt, col='0', row='variable', sharey='row')
    g.map_dataframe(sns.barplot, 'model', 'value', color='#25586E')#"Blues_d"
    g.set_titles(col_template="")
    g.set_xlabels('')
    g.set_ylabels('')
    g.set_axis_labels('')
    g.set_titles('')
    axes = g.axes.flatten()

    axes[0].set_ylabel('m / n')
    axes[7].set_ylabel('MR')
    axes[14].set_ylabel('MRR')
    axes[21].set_ylabel('Hits at 10')
    axes[28].set_ylabel('CRMRR')

    axes[0].set_title('web-google', loc='left', fontsize=18)
    axes[1].set_title('web-amazon', loc = 'left', fontsize = 18)
    axes[2].set_title('sx-stackoverflow', loc = 'left', fontsize = 18)
    axes[3].set_title('web-facebook', loc = 'left', fontsize = 18)
    axes[4].set_title('Industry_KG', loc = 'left', fontsize = 18)
    axes[5].set_title('BTC_alpha', loc = 'left', fontsize = 18)
    axes[6].set_title('BTC_otc', loc = 'left', fontsize = 18)


    axes[0].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.13700,edgecolor='#A2CB3D', fill='#A2CB3D',lw=6))
    axes[7].add_patch(plt.Rectangle((0.6, 0), 0.8, 49.3700,edgecolor='gray', fill='gray',lw=6))
    axes[14].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.02400,edgecolor='gray', fill='gray',lw=6))
    axes[21].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.36200,edgecolor='gray', fill='gray',lw=6))
    axes[28].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.38900,edgecolor='#A2CB3D', fill='#A2CB3D',lw=6))

    axes[1].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.19580,edgecolor='#A2CB3D', fill='#A2CB3D',lw=6))
    axes[8].add_patch(plt.Rectangle((1.6, 0), 0.8, 508.48000,edgecolor='#A2CB3D', fill='#A2CB3D',lw=6))
    axes[15].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.00900,edgecolor='gray', fill='gray',lw=6))
    axes[22].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.11200,edgecolor='gray', fill='gray',lw=6))
    axes[29].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.36800,edgecolor='#A2CB3D', fill='#A2CB3D',lw=6))

    axes[2].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.16550,edgecolor='#A2CB3D', fill='#A2CB3D',lw=6))
    axes[9].add_patch(plt.Rectangle((0.6, 0), 0.8, 38.40000,edgecolor='gray', fill='gray',lw=6))
    axes[16].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.02500,edgecolor='gray', fill='gray',lw=6))
    axes[23].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.49800,edgecolor='gray', fill='gray',lw=6))
    axes[30].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.28500,edgecolor='#A2CB3D', fill='#A2CB3D',lw=6))

    axes[3].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.09860,edgecolor='#A2CB3D', fill='#A2CB3D',lw=6))
    axes[10].add_patch(plt.Rectangle((1.6, 0), 0.8, 795.67000,edgecolor='gray', fill='#A2CB3D',lw=6))
    axes[17].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.00100,edgecolor='gray', fill='gray',lw=6))
    axes[24].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.30200,edgecolor='gray', fill='gray',lw=6))
    axes[31].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.13600,edgecolor='#A2CB3D', fill='#A2CB3D',lw=6))

    axes[4].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.11910,  edgecolor='#A2CB3D', fill='#A2CB3D', lw=6))
    axes[11].add_patch(plt.Rectangle((0.6, 0), 0.8, 54.85400,  edgecolor='gray', fill='gray', lw=6))
    axes[18].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.01800,  edgecolor='gray', fill='gray', lw=6))
    axes[25].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.57800,  edgecolor='gray', fill='gray', lw=6))
    axes[32].add_patch(plt.Rectangle((0.6, 0), 0.8, 0.28300,  edgecolor='#A2CB3D', fill='#A2CB3D', lw=6))

    axes[5].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.23510,  edgecolor='#A2CB3D', fill='#A2CB3D', lw=6))
    axes[12].add_patch(plt.Rectangle((1.6, 0), 0.8, 404.90000,  edgecolor='#A2CB3D', fill='#A2CB3D', lw=6))
    axes[19].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.00400,  edgecolor='gray', fill='gray', lw=6))
    axes[26].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.07600,  edgecolor='gray', fill='gray', lw=6))
    axes[33].add_patch(plt.Rectangle((1.6, 0), 0.8, 0.11900,  edgecolor='gray', fill='gray', lw=6))

    axes[6].add_patch(plt.Rectangle((-0.4, 0), 0.8, 0.15480,  edgecolor='#A2CB3D', fill='#A2CB3D', lw=6))
    axes[13].add_patch(plt.Rectangle((-0.4, 0), 0.8, 3.82000,  edgecolor='gray', fill='gray', lw=6))
    axes[20].add_patch(plt.Rectangle((-0.4, 0), 0.8, 0.11300,  edgecolor='#A2CB3D', fill='#A2CB3D', lw=6))
    axes[27].add_patch(plt.Rectangle((-0.4, 0), 0.8, 0.62600,  edgecolor='#A2CB3D', fill='#A2CB3D', lw=6))
    axes[34].add_patch(plt.Rectangle((-0.4, 0), 0.8, 0.15500,  edgecolor='gray', fill='gray', lw=6))

    plt.savefig(FP_FIG + '/FIG_' + str(todays_date) + '_rank-based_metrics.svg')
    plt.savefig(FP_FIG + '/FIG_' + str(todays_date) + '_rank-based_metrics.png')

    plt.show(block=True)
    plt.interactive(False)

# %% Plt the penalty term behavour.
def plot_penalty_term_behavior():
    todays_date = start()
    n = 100
    n_r = 10
    r = list(range(1, n_r + 1))
    # Total number of clusters.
    n_c = n_r
    c = list(range(1, n_c + 1))
    # filling the heatmap, value by value
    mat_crmrr = np.empty((n_r, n_c))
    for i in range(1, n_r):
        for j in range(1, n_c):
            mat_crmrr[i][j] = cluster_robust_mean_reciprocal_rank_penalty(r[i], c[j], n)
            # Pot the penalty term behaviour.
            plt.rc('font', family='serif')
            plt.rc('xtick', labelsize='x-small')
            plt.rc('ytick', labelsize='x-small')
            # Resize the matrix.
            x_lim = n_r + 1
            y_lim = x_lim
            # Plot the penalty as heatmap.
            ax = sns.heatmap(mat_crmrr,
                             annot=True,
                             annot_kws={"size": 11},
                             fmt=".2f",
                             # xticklabels = False,
                             # yticklabels = False,
                             cbar=False,
                             cmap="crest")
            ax.set(ylabel="$r_{ j | cc}$", xlabel="$c$")
            ax.set_xlim(0, x_lim)
            ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            ax.set_ylim(0, y_lim)
            ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            ax.set(title=r'Heatmap of the CRMRR penalty term, n = ' + str(n) + '.')
            ax.invert_yaxis()
            plt.savefig(
                FP_FIG + 'FIG_' + str(todays_date) + '_cluster_robust_mean_reciprocal_rank_penalty_term_n_' + str(n) + '.svg')
            plt.show()
            plt.show(block=True)
            plt.interactive(False)
### ---------------------------------------------------------------------------
### End.
#### ==========================================================================