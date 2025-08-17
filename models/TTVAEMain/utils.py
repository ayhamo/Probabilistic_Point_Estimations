from ttvae.main import train as train_ttvae
from ttvae.main import sample as sample_ttvae


import argparse

def execute_function(method, mode):
    if method == 'vae':
        mode = 'train'

    main_fn = eval(f'{mode}_{method}')

    return main_fn

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='tabsyn', help='Method: tabsyn or baseline.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')

    # configs for TTVAE
    parser.add_argument('--ttvae_epochs', type=int, default=100, help='Number of training epochs for TTVAE')
    parser.add_argument('--lsi_method', type=str, default='triangle', help='Latent space interpolation method')

    # configs for sampling in general
    parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data.')
    parser.add_argument('--steps', type=int, default=50, help='NFEs.')

    args = parser.parse_args()

    return args