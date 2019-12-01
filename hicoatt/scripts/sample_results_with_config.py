#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser
from hicoatt.configuration import Configuration
from hicoatt.dataset import load_json_from
from hicoatt.visualization import visualize_image_attention_with_config
import numpy as np


def main():
    parser = ArgumentParser("Start the model using the configuration.ini")
    parser.add_argument("-r", "--result_file", help="Path to the human readable results file", required=True)
    parser.add_argument("-m", "--model_path", help="", required=True)
    parser.add_argument("-b", "--batch_size", default=9, type=int, help="") 
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    
    run_opts = parser.parse_args()
    
    if run_opts.configuration:
        config = Configuration(run_opts.configuration)
    else:
        config = Configuration()
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.getGpuDevices())
    
    result_file_path = run_opts.result_file
    results = load_json_from(result_file_path)
    
    batch_size = run_opts.batch_size
    samples = np.random.choice(results, batch_size, replace=False)
    
    sqr_batch_size = np.sqrt(batch_size).astype("uint8")
    visualize_image_attention_with_config(run_opts.model_path, "word", samples, sqr_batch_size, sqr_batch_size, config)

        
if __name__ == '__main__':
    main()
    
