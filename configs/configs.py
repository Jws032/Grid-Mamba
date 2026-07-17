import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Event Point Cloud Segmentation')
    parser.add_argument('--config', default='configs/evisseg_evuav.yaml',type=str, help='path to config file')
    parser.add_argument('--runtime-only', action='store_true',
                        help='Measure inference runtime without writing predictions.txt or metrics.')
    parser.add_argument('--runtime-json', default=None, type=str,
                        help='Path to write runtime_summary.json.')
    parser.add_argument('--limit-test', default=None, type=int,
                        help='Optional number of test samples to process.')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)
    return args_cfg

cfg = get_parser()

