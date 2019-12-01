#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser
from hicoatt.configuration import Configuration
from hicoatt.training import start_training, start_prediction
from hicoatt.scripts import OPTION_DRY_RUN, OPTION_FORCE_SAVE
from hicoatt.attentions import start_extraction

def main():
    parser = ArgumentParser("Start the model using the configuration.ini")
    parser.add_argument("command", help="""One of [training, predict, extract]. 
                        training: Start training with the configuration.
                        predict: Apply a model on the dataset with the configuration and write the result file.
                        extract: Use a trained model and extract attention maps given a source and a target split.""")
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    parser.add_argument("-m", "--path_to_model", help="The absolute path to the model to predict or continue training.")
    parser.add_argument("-i", "--initial_epoch", type=int, help="The initial epoch to use when continuing training. This is required for continuing training.")
    parser.add_argument("-s", "--split_name", help="""The split name to perform the prediction or training on. This is required for predict. 
                        One of [train, validate, test_dev, test, trainval].
                        The split name determines sub-directories are files to lookup in the dataset directory.
                        A special split name is 'trainval' which combines both train and validate splits, then no validation is performed during training.
                        Notice on training: When the split name is not specified for training then the standard split of train and validate will be used. 
                        This is usually the wanted behavior. It is not possible for now to combine e.g. train and test.
                        Notice on predict: The split name is given with source and target split like 'trainval test_dev'
                        """)
    parser.add_argument("-d", "--dryrun", action="store_true")
    parser.add_argument("-f", "--force_save", action="store_true")
    
    run_opts = parser.parse_args()
    
    if run_opts.configuration:
        config = Configuration(run_opts.configuration)
    else:
        config = Configuration()
    config[OPTION_DRY_RUN] = run_opts.dryrun
    config[OPTION_FORCE_SAVE] = run_opts.force_save
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.getGpuDevices())
    
    config.dump()
        
    if run_opts.command == "training":
        start_training(config, run_opts.path_to_model, run_opts.initial_epoch, run_opts.split_name)
    
    if run_opts.command == "predict":
        if not run_opts.path_to_model:
            raise Exception("Cannot predict, when not model path is given. Please provide the path to the model and retry.")
        if not run_opts.split_name:
            raise Exception("Cannot predict, when no split name is given. Please provide the split name and retry.")
        source_split, target_split = run_opts.split_name.split(" ")
        if not source_split and target_split:
            print("Error: Cannot predict, when not both source and target split name are given: {}".format(run_opts.split_name))
            raise Exception("Please provide the split name like '-s train test' and retry.")
        
        start_prediction(config, run_opts.path_to_model, source_split, target_split)
    
    if run_opts.command == "extract":
        if not run_opts.path_to_model:
            raise Exception("Cannot extract, when not model path is given. Please provide the path to the model and retry.")
        if not run_opts.split_name:
            raise Exception("Cannot extract, when no split name is given. Please provide the split name and retry.")
        source_split, target_split = run_opts.split_name.split(" ")
        if not source_split and target_split:
            print("Error: Cannot extract, when not both source and target split name are given: {}".format(run_opts.split_name))
            raise Exception("Please provide the split name like '-s train test' and retry.")
        
        start_extraction(config, run_opts.path_to_model, source_split, target_split)

        
if __name__ == '__main__':
    main()
    
