### --------------------------------------------------------------------------
### Preamble.
### --------------------------------------------------------------------------
'''
This section is for CONSTANTS (in capital letters if unchanged).
'''
#%%
import os
### ---------------------------------------------------------------------------
#%% Filepaths.
### ---------------------------------------------------------------------------
# Define filepaths.
FP_ROOT = os.getcwd()
FP_FIG = FP_ROOT+'/figures/'
FP_DATA_IN = FP_ROOT+'/data/input/'
FP_MODELS = FP_ROOT+'/models/'
FP_DATA_OUT = FP_ROOT+'/data/output/'
# Initialize dataframe for rank-based metrics.
RB_METS = ['arithmetic_mean_rank', 'inverse_arithmetic_mean_rank', 'hits_at_10']
# Declare benchmark models.
models_str = ['TransR', 'DistMult', 'ComplEx']
# Indicate whether to include real-world industry data or not.
INCLUDE_INDUSTRY = False
# Indicate whether to include benchmark data or not.
INCLUDE_BENCHMARK = True
# Indicate whether to sample the data or not.
SAMPLE = True
# Specify sample ratio.
SAMPLE_RATIO = 0.001
# Specify training ratio.
EPOCHS = 5
BATCH_SIZE = 32
K = 10
# Plots.
PLOTS = False
### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------