import json
import argparse
from trainer import train
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    args = setup_parser().parse_args()
    args = vars(args)

    param = load_json(args['config'])
    # args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil_cifar.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
