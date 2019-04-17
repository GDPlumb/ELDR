
import sys
sys.path.insert(0, "../")
from vertices import run

data_file = "bipolar.tsv"
input_dim = 100
model_file = "model/model/perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_5400.ckpt"
num_clusters = 7

run(data_file, input_dim, model_file, num_clusters)
