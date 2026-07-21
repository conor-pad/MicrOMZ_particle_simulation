# main.py
import os
import warnings
import logging
import pickle

os.environ['KMP_DUPLICATE_LIB_OK']          = 'TRUE'
os.environ['TORCH_LOGS']                     = 'recompiles'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']    = '1'
warnings.filterwarnings('ignore', message='.*resized since it had shape.*')
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torch._dynamo').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import config as cfg
from physics import setup_physics
from loop import run_simulation
from plotting import generate_plots


def main():
    device, state = setup_physics(cfg)

    cache_file  = 'simulation_cache.pkl'
    force_rerun = True   # Set False to load previous run instantly

    if os.path.exists(cache_file) and not force_rerun:
        print(f'\n🚀 Loading cached simulation data from {cache_file}...')
        with open(cache_file, 'rb') as f:
            results = pickle.load(f)
    else:
        results = run_simulation(state, cfg, device)
        print(f'💾 Saving raw data to {cache_file}...')
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)

    generate_plots(*results, cfg)
    print('All processes complete!')


if __name__ == '__main__':
    main()
