### --------------------------------------------------------------------------
### Preamble.
### --------------------------------------------------------------------------
'''
This section is for Knowledge Graph Completion.
'''
#%%
import time
import pandas as pd
import numpy as np
from pykeen.models import DistMult
from pykeen.models import ComplEx
from pykeen.models import TransR
from constants import FP_DATA_OUT, RB_METS, EPOCHS, BATCH_SIZE, K, models_str
from utils import RankbasedMetricsandTopK, mean_rank, mean_reciprocal_rank
from cluster_robust_mean_reciprocal_rank import cluster_robust_mean_reciprocal_rank
### --------------------------------------------------------------------------
### Methods.
### --------------------------------------------------------------------------
#%%
def predict_links(models_str, lst_kgs, lst_kg_names, list_cluster_count):
    ### ---------------------------------------------------------------------------
    ### Declare variables.
    ### ---------------------------------------------------------------------------
    df_rb_mets = pd.DataFrame(RB_METS)
    # Create list of models.
    models = [TransR, DistMult, ComplEx]
    models_str = models_str
    # Specify epochs and batch size.
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    k = K
    ### ---------------------------------------------------------------------------
    ### Knowledge Graph Completion with clustered KGs and CRMRR.
    ### ---------------------------------------------------------------------------
    # Run the function for top k predictions for each dataset (loop is too expensive).
    for idx, graph in enumerate(lst_kgs):
        lst_graphs = [graph]
        lst_graph_str = [lst_kg_names[idx]]
        print(str(lst_kg_names[idx]))
        vars()['df_rb_mets_' + str(lst_kg_names[idx])], \
        vars()['df_top_k_triplets_' + str(lst_kg_names[idx])] = RankbasedMetricsandTopK(
            lst_graphs, lst_graph_str, models, models_str, epochs, batch_size, k, RB_METS, df_rb_mets)
        # Save top k triplets.
        vars()['df_top_k_triplets_' + str(lst_kg_names[idx])].to_csv(FP_DATA_OUT +
                                                                     't_Top_' + str(k) +
                                                                     '_triplets' + str(lst_kg_names[idx]) +
                                                                     '.csv', index=False)
        vars()['df_top_k_triplets_' + str(lst_kg_names[idx])] = pd.read_csv(FP_DATA_OUT +
                                                                            't_Top_' + str(k) +
                                                                            '_triplets' +
                                                                            str(lst_kg_names[idx]) +
                                                                            '.csv')
    print('---------------------------------------------------------------- \n' +
          'Finished conventional rank based metrics and top k on clustered lst_kgs at: ' +
          time.strftime('%H:%M:%S', time.gmtime(time.time())) +
          '. \n' +
          '----------------------------------------------------------------')
    ### Get partition of Intra-Cluster (IC) and Cross-Cluster (CC) predictions per top k df.
    list_n = []
    list_c = list_cluster_count
    list_ic_cc = []
    list_mr = []
    list_mrr = []
    list_hk = []
    list_crmrr = []
    for i, j in zip(lst_kg_names, list_cluster_count):
        # Define dataframes.
        df_pred = vars()['df_top_k_triplets_' + str(i)]
        df_clus = vars()['df_node_attributes_' + str(i)]
        # Create dictionary.
        dict_cluster_memberships = df_clus['c_CLUSTER']  # !!! Check here. df_clus.set_index('id')['c_CLUSTER']
        # Create columns for head respectively tail label cluster membership.
        df_pred['c_tail_id_cluster'] = df_pred['tail_id'].map(dict_cluster_memberships)
        df_pred['c_head_id_cluster'] = df_pred['head_id'].map(dict_cluster_memberships)
        # Assign IC /CC.
        df_pred['c_IC_CC'] = np.where(df_pred['c_tail_id_cluster'] == df_pred['c_head_id_cluster'],
                                      'IC',
                                      'CC')
        # Compute partition.
        print(df_pred['c_IC_CC'].value_counts(normalize=True, dropna=False))
        # Get count of clusters c.
        c = j
        # Get count of predictions n.
        n = len(df_pred['score'])
        list_n.append(n)
        # Compute IC/CC partition.
        ic_cc = df_pred['c_IC_CC'].value_counts()['CC'] / len(df_pred['c_IC_CC'])
        list_ic_cc.append(ic_cc)
        # Use the function from above to compute CRMRR.
        mr = mean_rank(df_pred['score'])
        list_mr.append(mr)
        # !!! Hits at k.
        mrr = mean_reciprocal_rank(df_pred['score'])
        list_mrr.append(mrr)
        # Get scores for IC and CC predictions.
        scores_ic = df_pred.loc[df_pred["c_IC_CC"] == "IC", "score"]
        scores_cc = df_pred.loc[df_pred["c_IC_CC"] == "CC", "score"]
        crmrr = cluster_robust_mean_reciprocal_rank(scores_ic, scores_cc, c, n)
        print('CMRR is of ' + str(i) + ': ' + str(crmrr))
        list_crmrr.append(crmrr)
    print('---------------------------------------------------------------- \n' +
          'Finished computing CRMRR at: ' +
          time.strftime('%H:%M:%S', time.gmtime(time.time())) +
          '. \n' +
          '----------------------------------------------------------------')
    ### ---------------------------------------------------------------------------
    ### Print table of rank-based metrics together with CRMRR.
    ### ---------------------------------------------------------------------------
    # Initialize dataframe for rank-based metrics with CRMRR
    # !!! Include hits at k.
    cols = ['G', 'n', 'c', 'CC/n', 'MR', 'MRR', 'CRMRR']
    # Create dataframe
    df_rb_mets_updated = pd.DataFrame(list(zip(lst_kg_names,
                                               list_n,
                                               list_c,
                                               list_ic_cc,
                                               list_mr,
                                               list_mrr,
                                               # list_hk,
                                               list_crmrr)), columns=cols)
    # Print table of updated rank based metrics.
    print(df_rb_mets_updated.to_latex(escape=False))
    ### ---------------------------------------------------------------------------
    ### Save files.
    ### ---------------------------------------------------------------------------
    # Save metrics.
    df_rb_mets_updated.to_csv(FP_DATA_OUT + 't_rb_mets.csv', index=False)
    # TAB_Rankbased_metrics.csv
    df_rb_mets_updated = pd.read_csv(FP_DATA_OUT + 't_rb_mets.csv')
    print('---------------------------------------------------------------- \n' +
          'Finished saving datasets at: ' +
          time.strftime('%H:%M:%S', time.gmtime(time.time())) +
          '. \n' +
          '----------------------------------------------------------------')
    return df_rb_mets_updated