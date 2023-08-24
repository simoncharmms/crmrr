### --------------------------------------------------------------------------
### Preamble.
### --------------------------------------------------------------------------
'''
This section is for loading datasets.
'''
#%%
import pickle
import time
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg as la
from pykeen.models import DistMult
from pykeen.models import ComplEx
from pykeen.models import TransR
from constants import SAMPLE, SAMPLE_RATIO, FP_DATA_IN, FP_DATA_OUT, RB_METS, EPOCHS, BATCH_SIZE, K, \
    INCLUDE_BENCHMARK, INCLUDE_INDUSTRY
from utils import decode_graph, dataframe_to_knowledge_graph, \
    RankbasedMetricsandTopK, mean_rank, mean_reciprocal_rank
from cluster_robust_mean_reciprocal_rank import cluster_robust_mean_reciprocal_rank
### --------------------------------------------------------------------------
### Methods.
### --------------------------------------------------------------------------
#%% Get bitcoin benchmark dataset.
def get_bitcoin_benchmark_kg():
    df_bitcoin_alpha = pd.read_csv(
        FP_DATA_IN + 'bitcoinalpha.csv',
        skiprows=1,
        names=['head', 'tail', 'weight', 'timestamp'])
    df_bitcoin_otc = pd.read_csv(
        FP_DATA_IN + 'soc_sign_bitcoinotc.csv',
        skiprows=1,
        names=['head', 'tail', 'weight', 'timestamp'])
    return df_bitcoin_alpha, df_bitcoin_otc
# Get facebook benchmark dataset.
def get_facebook_benchmark_kg():
    FP = FP_DATA_IN +'facebook_large/musae_facebook_target.csv'
    # read target file
    df_target = pd.read_csv(FP)
    FP = FP_DATA_IN +'facebook_large/musae_facebook_edges.csv'
    # read edge file
    df_edge = pd.read_csv(FP)
    # Make dictionary.
    dict = df_target.set_index('id').T.to_dict()
    df_facebook = df_edge
    df_facebook['weight'] = df_edge['id_1'].map(dict)
    df_facebook = df_facebook.rename(columns={'id_1': 'head', 'id_2': 'tail'})
    df_facebook['timestamp'] = ''
    return df_facebook
# Get amazon benchmark dataset.
def get_amazon_benchmark_kg():
    FP = FP_DATA_IN +'com-amazon.ungraph.txt'
    # read text file
    with open(FP, 'r') as f:
        df = f.read()
    # split the data into rows
    rows = df.split('\n')
    # remove the first two rows
    rows = rows[4:]
    # create a list of tuples containing the data in each row
    data_list = []
    for row in rows:
        try:
            row_data = row.split('\t')
            data_list.append((row_data[0], row_data[1]))
        except:
            pass
    # create a dataframe from the list of tuples
    df_amazon = pd.DataFrame(data_list, columns=['head', 'tail'])
    df_amazon['weight'] = ''
    df_amazon['timestamp'] = ''
    return df_amazon
# Get stackoverflow benchmark dataset.
def get_stackoverflow_benchmark_kg():
    FP = FP_DATA_IN +'sx-stackoverflow.txt'
    # read text file
    with open(FP, 'r') as f:
        df = f.read()
    # split the data into rows
    rows = df.split('\n')
    # create a list of tuples containing the data in each row
    data_list = []
    for row in rows:
        try:
            row_data = row.split(' ')
            data_list.append((row_data[0], row_data[1], row_data[2]))
        except:
            pass
    # create a dataframe from the list of tuples
    df_stackoverflow = pd.DataFrame(data_list, columns=['head', 'tail', 'weight'])
    df_stackoverflow['timestamp'] = ''
    df_stackoverflow = df_stackoverflow.sample(frac=0.01, replace=True, random_state=1)
    return df_stackoverflow
# Get google benchmark dataset.
def get_google_benchmark_kg():
    FP = FP_DATA_IN +'web-Google.txt'
    # read text file
    with open(FP, 'r') as f:
        df = f.read()
    # split the data into rows
    rows = df.split('\n')
    # remove the first two rows
    rows = rows[4:]
    # create a list of tuples containing the data in each row
    data_list = []
    for row in rows:
        try:
            row_data = row.split('\t')
            data_list.append((row_data[0], row_data[1]))
        except:
            pass
    # create a dataframe from the list of tuples
    df_google = pd.DataFrame(data_list, columns=['head', 'tail'])
    df_google['weight'] = ''
    df_google['timestamp'] = ''
    return df_google
#%% Wrapper function.
def get_data():
    # List for KGs.
    lst_kgs = []
    # List for KG dataframes.
    lst_kg_dfs = []
    # List for KG dataframe names.
    lst_kg_names = []
    # Specify the graph nomenclature.
    head = 'head'
    relation = 'is_related_to'
    tail = 'tail'
    attributes = ['weight', 'timestamp']
    # If clauses.
    if INCLUDE_INDUSTRY:
        lst_industry_kg_names = ['Industry_KG']
        with open(FP_DATA_IN+'industryKG.gpickle', 'rb') as f:
            # Load the graph.
            G_Industry_KG = pickle.load(f)
            # Append to list.
            lst_kgs.append(G_Industry_KG)
            # Make edgelist.
            df_Industry_KG = nx.to_pandas_edgelist(G_Industry_KG)
            # Adjust column names.
            df_Industry_KG = df_Industry_KG.rename(columns={'source': 'head',
                                                            'target': 'tail',
                                                            'c_VOLUME_OVER_LIFETIME' : 'weight',
                                                            'c_SMP_TIMESTAMP' : 'timestamp'})
            # # Append to list.
            # lst_kg_dfs.append(df_Industry_KG)
        print('Finished loading industry knowledge graph at: ' +
              time.strftime('%H:%M:%S', time.gmtime(time.time())) + '. \n' +
              '----------------------------------------------------------------- \n')
    ### ---------------------------------------------------------------------------
    ### Load benchmark datasets.
    ### ---------------------------------------------------------------------------
    # os.chdir(fp_bench)
    '''
    This is who-trusts-whom network of people who trade using Bitcoin on a platform 
    called Bitcoin OTC. Since Bitcoin users are anonymous, there is a need to 
    maintain a record of users' reputation to prevent transactions with fraudulent 
    and risky users. Members of Bitcoin OTC rate other members in a scale of 
    -10 (total distrust) to +10 (total trust) in steps of 1. 
    This is the first explicit weighted signed directed network available for research.
    The second dataset is the same for the Bitcoin platform OTC.
    '''
    # Create benchmark KGs.
    if INCLUDE_BENCHMARK:
        # Specify names.
        lst_bench_kg_names = ['BTC_alpha',
                              'BTC_otc',
                              'web-google',
                              'sx-stackoverflow',
                              'web-amazon',
                              'web-facebook']
        # Read the first benchmark dataset.
        df_bitcoin_alpha, df_bitcoin_otc = get_bitcoin_benchmark_kg()
        lst_kg_dfs.append(df_bitcoin_alpha)
        lst_kg_dfs.append(df_bitcoin_otc)
        # Get Google benchmark KG.
        df_google = get_google_benchmark_kg()
        lst_kg_dfs.append(df_google)
        # Get Stackoverflow benchmark KG.
        df_stackoverflow = get_stackoverflow_benchmark_kg()
        lst_kg_dfs.append(df_stackoverflow)
        # Get Amazon benchmark data.
        df_amazon = get_amazon_benchmark_kg()
        lst_kg_dfs.append(df_amazon)
        # Get Facebook benchmark data.
        df_facebook = get_facebook_benchmark_kg()
        lst_kg_dfs.append(df_facebook)
        print('----------------------------------------------------------------- \n' +
              'Finished loading benchmark datasets at: ' +
              time.strftime('%H:%M:%S', time.gmtime(time.time())) +
              '. \n' +
              '-----------------------------------------------------------------')
    # Sample if required. The industry KG is not sampled (there is no df, it is only in list_kgs).
    if SAMPLE:
        # Create sampled list.
        lst_kg_dfs_sampled = []
        for df in lst_kg_dfs:
            # Sample dataframe.
            lst_kg_dfs_sampled.append(df.sample(frac=SAMPLE_RATIO, random_state=1))
        # Overwrite original list-
        lst_kg_dfs = lst_kg_dfs_sampled
    # Crate list of KG names.
    if INCLUDE_INDUSTRY and INCLUDE_BENCHMARK:
        lst_kg_names = [*lst_industry_kg_names, *lst_bench_kg_names]
    elif INCLUDE_INDUSTRY:
            lst_kg_names = lst_industry_kg_names
    elif INCLUDE_BENCHMARK:
            lst_kg_names = lst_bench_kg_names
    # Create list of KGs as nx graph object also for benchmark graphs.
    try:
        for idx, source_df in enumerate(lst_kg_dfs):
            name = lst_bench_kg_names[idx]
            # Create KGs for each benchmark dataframe. The industry KG is already a graph object.
            vars()['G_' + name] = dataframe_to_knowledge_graph(head, relation, tail, attributes, source_df)
            # Save raw KG.
            nx.write_gml(vars()['G_' + name], FP_DATA_OUT + 'G_' + name + '.gml')
            lst_kgs.append(vars()['G_' + name])
    except: pass
    # Name graph metrics for table.
    mets = ['Node count', 'Relation count', 'Mean degree', 'Transitivity', 'Mean clustering coefficent']
    # Create dataframe
    df_graph_mets = pd.DataFrame(mets)
    # Loop through KG dataframes.
    for idx, graph in enumerate(lst_kgs):
        name = lst_kg_names[idx]
        # Print the metrics of the KG.
        vars()[name+'_mets'] = decode_graph(vars()['G_'+ name])
        # Add the list of metrics to the metrics dataframe.
        df_graph_mets[name] = vars()[name+'_mets']
        print('----------------------------------------------------------------- \n' +
              'Finished decoding '+ str(name) +' at: ' +
              time.strftime('%H:%M:%S', time.gmtime(time.time())) +
              '. \n' +
              '-----------------------------------------------------------------')
    # Print graph metrics for LaTeX.
    print(df_graph_mets.to_latex(escape=False))
    # Save graph metrics.
    df_graph_mets.to_csv(FP_DATA_OUT + 't_Graph_metrics.csv', index=False)
    df_graph_mets = pd.read_csv(FP_DATA_OUT + 't_Graph_metrics.csv')
    print('----------------------------------------------------------------- \n' +
          'Finished decoding datasets at: ' +
          time.strftime('%H:%M:%S', time.gmtime(time.time())) +
          '. \n' +
          '-----------------------------------------------------------------')
    return df_graph_mets, lst_kg_names, lst_kgs
# Compute benchmark dataset unrealistic completion candidate share.
#%%
def train_models(lst_kgs, lst_kg_names):
    ### ---------------------------------------------------------------------------
    ### Declare variables.
    ### ---------------------------------------------------------------------------
    df_rb_mets = pd.DataFrame(RB_METS)
    # Create list of models.
    models = [TransR, DistMult, ComplEx]
    models_str = ['TransR', 'DistMult', 'ComplEx']
    # Specify epochs and batch size.
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    k = K
    ### ---------------------------------------------------------------------------
    ### Leiden Algorithm clustering.
    ### ---------------------------------------------------------------------------
    # Create empty list for storing count of clusters per graph.
    list_cluster_count = []
    for graph, i in zip(lst_kgs, lst_kg_names):
        # Convert networkx graph to igraph.
        igraph = ig.Graph.from_networkx(graph)
        partition = la.find_partition(igraph, la.ModularityVertexPartition)
        cluster_membership = partition.membership
        # Add the cluster attribute to the graph.
        # nx.set_node_attributes(graph, cluster_membership, "c_CLUSTER")
        vars()['df_node_attributes_' + str(i)] = pd.DataFrame(
            {'id': list(graph.nodes()), 'c_CLUSTER': cluster_membership}
        )
        # Append the maximum number of clusters to a list.
        c = int(len(vars()['df_node_attributes_' + str(i)]['c_CLUSTER'].unique()))
        list_cluster_count.append(c)
        node_attr = vars()['df_node_attributes_' + str(i)].set_index('id').to_dict('index')
        nx.set_node_attributes(graph, node_attr)
        print(str(i) + ' has ' + str(c) + ' clusters.')
        # Save clustered KG.
        nx.write_gml(graph, FP_DATA_OUT + 'G_' + i + '_clustered.gml')
    print('---------------------------------------------------------------- \n' +
          'Finished clustering at: ' +
          time.strftime('%H:%M:%S', time.gmtime(time.time())) +
          '. \n' +
          '----------------------------------------------------------------')
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
                                                                     '_triplets'+str(lst_kg_names[idx])+
                                                                     '.csv', index=False)
        vars()['df_top_k_triplets_' + str(lst_kg_names[idx])] = pd.read_csv(FP_DATA_OUT +
                                                                            't_Top_' + str(k) +
                                                                            '_triplets'+
                                                                            str(lst_kg_names[idx])+
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
        dict_cluster_memberships = df_clus['c_CLUSTER'] #!!! Check here. df_clus.set_index('id')['c_CLUSTER']
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
    df_rb_mets_updated.to_csv(FP_DATA_IN + 't_Rankbased_metrics.csv', index=False)
    df_rb_mets_updated = pd.read_csv(FP_DATA_IN + 't_Rankbased_metrics.csv')
    print('---------------------------------------------------------------- \n' +
          'Finished saving datasets at: ' +
          time.strftime('%H:%M:%S', time.gmtime(time.time())) +
          '. \n' +
          '----------------------------------------------------------------')
    return df_rb_mets_updated
### --------------------------------------------------------------------------
### End.
### --------------------------------------------------------------------------

