#### ==========================================================================
#### Generate a temporal Knowledge Graph.
#### Author: Simon Schramm, AU-22.
#### Created on: 28.11.2022.
#### --------------------------------------------------------------------------
"""
This directory reads the relevant source datasets and creates an advanced
temporal Knowledge Graph.
"""
#%%
from src.get_data import get_data, train_models
from src.constants import PLOTS
from src.cluster_graphs import cluster_graphs
from src.predict_links import predict_links
from src.figures import plot_penalty_term_behavior, plot_kg_metrics, plot_rank_based_metrics
#%%
def main(models):
    df_graph_mets, lst_kg_names, lst_kgs = get_data()
    if PLOTS:
        plot_kg_metrics(df_graph_mets)
    lst_kgs, lst_kg_names, list_cluster_count = cluster_graphs(lst_kgs, lst_kg_names)
    df_rb_mets = predict_links(models, lst_kgs, lst_kg_names, list_cluster_count)
    if PLOTS:
        plot_penalty_term_behavior()
        plot_rank_based_metrics()
    return df_rb_mets
# Call the main function.
if __name__ == '__main__':
    result = main(
        models = ['TransR', 'DistMult', 'ComplEx']
    )
### ---------------------------------------------------------------------------
### End.
#### ==========================================================================