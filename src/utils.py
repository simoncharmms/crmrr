#### ==========================================================================
#### Apply cluster-robust inference to BMA AMM KG and benchmark knoledge graphs.
#### Author: Simon Schramm, AU-22.
#### Created on: 15.11.2022.
#### --------------------------------------------------------------------------
"""
This scripts implements the cluster-robust inference algorithm and
compares its performance on the BMW AMM KG to other benchmark KGs.
"""
### ---------------------------------------------------------------------------
#%% Preamble.
### ---------------------------------------------------------------------------
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import pandas as pd
import networkx as nx
import time
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
from pykeen import predict
import datetime
from constants import FP_MODELS
### --------------------------------------------------------------------------
#%% Pre- and Postscript.
### --------------------------------------------------------------------------
def start():
    global todays_date
    # blockPrint()
    # Getting todays date and time using now() of datetime
    current_date = datetime.datetime.now()
    # Get date only.
    todays_date = int(current_date.strftime("%Y%m%d"))
    # Get date and time.
    todays_date_and_time = int(current_date.strftime("%Y%m%d%H%M%S"))
    # Get start time of the script.
    start_time = time.time()
    # Print start metrics of this script.
    print('================================================================= \n' +
          'Script \n' + sys.argv[0] + '\n' + 'started at: ' +
          time.strftime('%H:%M:%S', time.gmtime(start_time)) + '. \n' +
          '-----------------------------------------------------------------')
    # Define next script for pipeline config.
    # next_script = 'akt_ ... .py'
    return todays_date

def stop(start_time):
    # enablePrint()
    elapsed_time = time.time() - start_time
    print('---------------------------------------------------------------- \n' +
          'Elapsed time: ' +
          time.strftime('%H:%M:%S', time.gmtime(elapsed_time)) +
          '. \n' +
          '================================================================')
    # os.chdir(project_path)
    # os.system(next_script)
### ---------------------------------------------------------------------------
#%% Functions.
### ---------------------------------------------------------------------------
'''
This section is purely for declaring functions.
'''
### ---------------------------------------------------------------------------
#%% Decode graph.
### ---------------------------------------------------------------------------
def dataframe_to_knowledge_graph(head, relation, tail, attributes, source_df):
    '''
    Create a networkx directed graph with attributes from a source dataframe.
    '''
    # Initialize the knowledge graph dataframe.
    kg_df = pd.DataFrame(
        {'source':source_df[head],
         'my_edge_key':relation,
         'target':source_df[tail]})
    # Iterate over attributes and add them to the dataframe as columns.
    for j in range(len(attributes)):
        kg_df[str(attributes[j])]=source_df[attributes[j]]
    kg_df['c_relation_label'] = relation
    # Create graph.
    G = nx.from_pandas_edgelist(kg_df,
                              edge_key = relation,
                              edge_attr = attributes,
                              create_using=nx.DiGraph())
    return G
### --------------------------------------------------------------------------
### Decode graph.
### --------------------------------------------------------------------------
#%% Decode the graph and print certain clustering parameters.
def decode_graph(G):
    # G.remove_nodes_from(nx.isolates(G))
    G_node_count = G.number_of_edges()
    print('Num of edges: {}'.format(G_node_count))
    G_rel_count = G.number_of_nodes()
    print('Num of nodes: {}'.format(G_rel_count))
    G_deg = nx.degree_histogram(G)
    G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
    G_deg_avg = sum(G_deg_sum) / G.number_of_nodes()
    print('Average degree: {:.2f}'.format(G_deg_avg))
    '''
    The diameter of a directed, weakly connected graph results in infinite path lengths.
    Thus, we find try a top down approach in finding the maximum distance from the shortest paths in the graph.
    Essentially, we find the maximum diameter of G, not the diameter of G itself.
    '''
    #G_max_diam = max([max(j.values()) for (i, j) in nx.shortest_path_length(G)])
    #print('Maximum diameter: {}'.format(G_max_diam))
    '''
    The transitivity of a graphs is the fraction of all possible triangles.
    Triangles are two edges with a shared relation.
    '''
    G_trans = nx.transitivity(G)
    print('Transitivity: {:.2f}'.format(G_trans))
    '''
    The clustering is the fraction of all possible directed triangles / 
    geometric average of the subgraph edge weights.
    '''
    G_cluster = sorted(nx.clustering(G).values())
    G_cluster_coef = sum(G_cluster) / len(G_cluster)
    print('Average clustering coefficient: {:.2f}'.format(sum(G_cluster) / len(G_cluster)))
    '''
    Modularity
    '''
    #G_mod_mat = nx.directed_modularity_matrix(G)
    #G_mod_mat_max = G_mod_mat.max()
    #print('Maximum of directed modularity matrix: {:.2f}'.format(G_mod_mat_max))
    #G_mets = [G_node_count, G_rel_count, G_deg_avg, G_max_diam, G_trans, G_cluster_coef, G_mod_mat_max]
    G_mets = [G_node_count, G_rel_count, G_deg_avg, G_trans, G_cluster_coef]
    return G_mets
### ---------------------------------------------------------------------------
### Determine unrealistic completion candidates.
### ---------------------------------------------------------------------------
def getrealisticpredictionsportion(edge_df, df_top_k_triplets):
    '''
    Realistic completion candidates are assumed to be those that can be inferred from
    the original edge dataframe. Realistic tail predictions are thus these, that
    contain tails from the list of unique tails of the respective relation of the
    original dataset. In a sense, this is a CWA for validation.
    '''
    # Initialize dataframe for realistic completion candidates.
    real_cc = ['type', 'target']
    # Create dataframe.
    df_real_cc = pd.DataFrame(real_cc)
    #!!! Hier brauchen wir was um die Cross Cluster Completion Candidates zu erkennen.
    # Iterate through original edge dataframe.
    for val in edge_df['type'].unique():
        df_edges = edge_df[edge_df['type']==val]
        df_edges = df_edges[real_cc]
        df_real_cc = pd.concat([df_real_cc, df_edges]).drop_duplicates()
        df_real_cc['c_check'] = df_real_cc['type'].astype('str') + df_real_cc['target'].astype('str')
    # Check whether top k predictions are realistic under CWA or not.
    df_top_k_triplets['c_check'] = df_top_k_triplets['relation_id'].astype('str') + df_top_k_triplets['tail_id'].astype('str')
    df_top_k_triplets['c_CWA_realistic_completion_candidate'] = np.where(df_top_k_triplets['c_check'].isin(df_real_cc['c_check']),
                                                                         'Realistic',
                                                                         'Unrealistic')
    print(df_top_k_triplets['c_CWA_realistic_completion_candidate'].value_counts(normalize=True, dropna=False))
    return df_top_k_triplets
### --------------------------------------------------------------------------
### Rank based metrics for embedding-based KGC.
### --------------------------------------------------------------------------
# Compute the mean rank.
def mean_rank(scores):
    '''
    "score" should be an array of those scores that correspond to true
    completion candidates --> "in_training" = True.
    '''
    ranks = scores.rank()
    return np.mean(ranks)
# Compute the mean reciprocal rank.
def mean_reciprocal_rank(scores):
    '''
    "score" should be an array of those scores that correspond to true
    completion candidates --> "in_training" = True.
    '''
    ranks = scores.rank()
    recip_ranks = np.reciprocal(ranks)
    return np.mean(recip_ranks)

### --------------------------------------------------------------------------
#%% Wrapper function for pipeline config.
### --------------------------------------------------------------------------
def RankbasedMetricsandTopK(lst_graphs, lst_graph_str, models, models_str, epochs, batch_size, k, RB_METS, df_rb_mets):
    # specifiy test, validation and training ratios.
    train = .7
    validation = .2
    test = .1
    # Initialize top-k triplets dataframe.
    print('----------------------------------------------------------------- \n' +
          'Starting rank-based metric and top k computation with \n' +
          'Train / Test / Validation: ' + str(train) + ', ' + str(test) + ', ' + str(validation) )
    df_top_k_triplets = pd.DataFrame()
    for graph, i in zip(lst_graphs, lst_graph_str):
        print('----------------------------------------------------------------- \n' +
              'Graph: ' + str(i) + '.\n')
        # Create test and training triples.
        edge_df = nx.to_pandas_edgelist(graph)
        # Set the relation label column as type-column for TriplesFactory.
        if 'c_relation_label' not in edge_df.columns:
            edge_df['c_relation_label'] = "has_relation"
        edge_df["type"] = edge_df['c_relation_label'].astype(str)
        edge_df["target"] = edge_df['target'].astype(str) # Important for numeric labels in BTC graphs!
        edge_df["source"] = edge_df['source'].astype(str)
        # Clean up the edge dataframe.
        edge_df_clean = edge_df[["source", "type", "target"]]
        edge_df_clean = edge_df_clean.dropna(thresh=3)
        edge_df_clean = edge_df_clean[edge_df_clean.type != 'nan']
        #  Note the automatically assigned random_state=1480565713.
        tf = TriplesFactory.from_labeled_triples(
            edge_df_clean[["source", "type", "target"]].values,
            create_inverse_triples=True,
            entity_to_id=None,
            relation_to_id=None,
            compact_id=False,
            filter_out_candidate_inverse_relations=True,
            metadata=None
        )
        # Split in train, test and validation.
        training_tf, testing_tf, validation_tf = tf.split([train, test, validation])
        # Use train split from triple factory.
        training_triples_factory = training_tf
        for mod, j in zip(models, models_str):
            print('Model: ' + str(j) + '.\n' +
                  '-----------------------------------------------------------------')
            # Pick a model and initial HPs.
            model = mod(triples_factory=training_triples_factory, random_seed = 1)
            # Saving the model is not required, we'll save the pipeline later.
            # Use Pykeen's built-in pipeline function. Duration on MacOS ~ 7 minutes.
            '''
            Stochastic Local Closed World Assumption and basic negative sampler for non-overlapping false positives.
            '''
            result = pipeline(
                training=training_tf,
                testing=testing_tf,
                validation=validation_tf,
                model=model,
                optimizer='Adam',
                stopper='early',
                evaluation_kwargs=dict(
                    use_tqdm=False,
                    batch_size=batch_size),
                random_seed=1,
                device='cpu',
                training_kwargs=dict(
                    num_epochs=epochs,
                    batch_size=batch_size,
                    #checkpoint_name=FP_MODELS+'MOD_'+str(j)+'_KGC_pipeline_checkpoint.pt',
                    #checkpoint_frequency=5,
                    use_tqdm_batch=False
                    )
                )
            # Save results of the pipeline.
            result.save_to_directory(FP_MODELS+'MOD_'+str(j)+'_KGC_pipeline_results')
            # Create the rank-based evaluator
            evaluator = RankBasedEvaluator()
            # Get rank-based results.
            '''
            We use the Bordes et al. (2013) approach to filtering.
            '''
            rankbasedmetrics = evaluator.evaluate(model=model,
                                                  mapped_triples=result.training.mapped_triples,
                                                  batch_size=batch_size,
                                                  additional_filter_triples=[
                                                      result.training.mapped_triples
                                                  ])
            vars()['df_' + str(i) + '_' + str(j) + '_mets'] = rankbasedmetrics.to_df()
            # Add to dataframe.
            df_rb_mets_temp = rankbasedmetrics.to_df()
            # Create a mask for filtering the rank-based metrics df.
            '''
            Pykeen evaluator provides head, relation and tail prediction metrics.
            We filter to tail for the KGC task and drill down to the realistic approximation.
            '''
            mask = (df_rb_mets_temp['Side'] == 'tail') & \
                   (df_rb_mets_temp['Type'] == 'realistic') & \
                   (df_rb_mets_temp['Metric'].isin(RB_METS))
            df_rb_mets_temp = df_rb_mets_temp[mask]
            df_rb_mets['c_' + str(i) + '_' + str(j)] = df_rb_mets_temp['Value'].tolist()
            # Get scores for top 15 triples
            #vars()['df_' + str(i) + '_' + str(j) + '_top_k_triplets'] = predict_target(model, triples_factory=result.training) #!!! Check here. Former: result.training
            vars()['df_' + str(i) + '_' + str(j) + '_top_k_triplets'] = predict.predict_all(model=model, k=k, batch_size=batch_size, target='tail').process().df
            '''
            Here: https://pykeen.readthedocs.io/en/stable/reference/predict.html
            the authors state that 
            predictions_df = predict.get_all_prediction_df(model, triples_factory=result.training)
            can be replaced by
            predict.predict_all(model=model, triples_factory=result.training).process().df
            but triples_factory is an unexpected keyword            
            '''
            vars()['df_' + str(i) + '_' + str(j) + '_top_k_triplets']['graph_model'] = str(i) + '_' + str(j)
            # Add a column with indicates whether, under CWA, a cc is realistic or not.
            vars()['df_' + str(i) + '_' + str(j) + '_top_k_triplets'] = getrealisticpredictionsportion(edge_df, vars()['df_' + str(i) + '_' + str(j) + '_top_k_triplets'])
            # Append top_k datasets and store.
            # !!! Appending does not work.
            df_top_k_triplets = pd.concat([df_top_k_triplets, vars()['df_' + str(i) + '_' + str(j) + '_top_k_triplets']])
    return df_rb_mets, df_top_k_triplets
print('---------------------------------------------------------------- \n' +
      'Finished functions at: ' +
      time.strftime('%H:%M:%S', time.gmtime(time.time())) +
      '. \n' +
      '----------------------------------------------------------------')
