import os

parent_dir = os.path.dirname(os.getcwd())
data_path = os.path.join(parent_dir, 'data')
log_path = os.path.join(parent_dir, 'data', 'driving_log.csv')
save_path = os.path.join(parent_dir, 'output')

TRAINING_RATIO = 0.8
NUM_EPOCHS = 22

loader_params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 1}


