### --------------------------------------------------------------------------
### Preamble.
### --------------------------------------------------------------------------
'''
This section is for clustering graphs.
'''
#%%
import time
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la
from pykeen.models import DistMult
from pykeen.models import ComplEx
from pykeen.models import TransR
from constants import FP_DATA_OUT, RB_METS, EPOCHS, BATCH_SIZE, K
### --------------------------------------------------------------------------
### Methods.
### --------------------------------------------------------------------------
#%%
def cluster_graphs(lst_kgs, lst_kg_names):
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
    return lst_kgs, lst_kg_names, list_cluster_count
### --------------------------------------------------------------------------
### End.
### --------------------------------------------------------------------------

