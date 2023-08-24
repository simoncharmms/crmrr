### --------------------------------------------------------------------------
### Preamble.
### --------------------------------------------------------------------------
'''
This section is for computing the cluster robust mean reciprocal rank.
'''
#%%
import pandas as pd
import numpy as np
### --------------------------------------------------------------------------
### Methods.
### --------------------------------------------------------------------------
#%%
#%% Compute the cluster-robust variants.
def cluster_robust_mean_reciprocal_rank_penalty(r, c, n):
    '''
    This is the penalty term of the Cluster Robust Mean Reciprocal Rank (CRMRR).
    '''
    return pow(c, -(r / n)) - 1
#%% Full Cluster Robust Mean Reciprocal Rank (CRMRR).
def cluster_robust_mean_reciprocal_rank(scores_ic, scores_cc, c, n):
    '''
    This is the Cluster Robust Mean Reciprocal Rank (CRMRR).
    "scores_ic" contains the scores true completion candidates within a cluster, while
    "scores_cc" contains the scores true completion candidates across clusters.
    '''
    # Get list of all scores.
    scores = pd.concat([scores_ic, scores_cc])
    # Get ranks.
    ranks = scores.rank()
    # Get in-cluster ranks.
    ranks_ic = ranks[:len(scores_ic)]
    # Get cross-cluster ranks.
    ranks_cc = ranks[len(scores_ic):]
    # Initialize list for reciprocal ranks.
    recip_ranks = []
    # Loop through intra-cluster ranks.
    for r in ranks_ic:
        recip_ranks.append(np.reciprocal(r))
    # Initialize list for penalized ranks.
    ranks_cc_pen = []
    # Loop through cross-cluster ranks and penalize.
    for r in ranks_cc:
        ranks_cc_pen.append(cluster_robust_mean_reciprocal_rank_penalty(r, c, len(ranks_ic+ranks_cc)))
    sum_recip_ranks = sum(recip_ranks)
    sum_ranks_cc_pen = sum(ranks_cc_pen)
    print('The sum of reciprocal ranks is: ' + str(sum_recip_ranks) +
          ' and the sum of penalties is: ' + str(sum_ranks_cc_pen))
    cr_recip_ranks = sum_recip_ranks + sum_ranks_cc_pen
    return (cr_recip_ranks / n)*(-1)
#%%

### --------------------------------------------------------------------------
### End.
### --------------------------------------------------------------------------

