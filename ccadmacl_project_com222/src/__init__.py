from . import _1_data_processing
from . import _2_model
from . import _3_train_model
from . import _4_evaluate_model
from . import pipeline 
from . import utils 

# Optionally export functions or classes for easy access
from ._1_data_preprocessing import load_data, preprocess_data
from ._2_model import train_kmeans, evaluate_clustering
from ._3_train_model import train_and_evaluate
from ._4_evaluate_model import predict_cluster
