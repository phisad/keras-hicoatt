#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser

from hicoatt.configuration import Configuration
from hicoatt.dataset import to_split_dir
from hicoatt.dataset.images import _get_image_paths
from hicoatt.dataset.images.preprocessing.preprocess import get_preprocessing_tfrecord_file, \
    preprocess_images_and_write_tfrecord
from hicoatt.dataset.images.feature_maps.distributed import create_many_feature_map_files_from_config
from hicoatt.scripts import OPTION_DRY_RUN
from hicoatt import SPLIT_VALIDATE, SPLIT_TRAIN

def main():
    parser = ArgumentParser("Prepare the VQA 1.0 dataset for training")
    parser.add_argument("command", help="""One of [preprocess, featuremaps, all]. 
                        preprocess: Resizes images and stores them by image id in a TFRecord file 
                        featuremaps: Loads images from the TFRecords file and creates the feature maps using the visual model.
                                     Then stores the feature maps as individual files along with the images.
                        all: All of the above""")
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    parser.add_argument("-d", "--dryrun", action="store_true")
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-n", "--num_images", type=int, help="The expected number of images in the TFRecord file. Will show a progress bar then.")
    parser.add_argument('-s', "--split_names", nargs='+', help="Specify the split names. Otherwise defaults to [train, validate]")
    run_opts = parser.parse_args()
    
    if run_opts.configuration:
        config = Configuration(run_opts.configuration)
    else:
        config = Configuration()
        
    config[OPTION_DRY_RUN] = run_opts.dryrun
    config["batch_size"] = run_opts.batch_size
    config["num_images"] = run_opts.num_images

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.getGpuDevices())
    
    split_names = [SPLIT_TRAIN, SPLIT_VALIDATE]
    if run_opts.split_names:
        split_names = run_opts.split_names

    print("Starting image preparation: {}".format(run_opts.command))
    directory_path = config.getDatasetImagesDirectoryPath()
    
    if run_opts.command in ["all", "preprocess"]:
        print("Perform preprocessing for splits: " + str(split_names))
        for split_name in split_names:
            tfrecord_file = get_preprocessing_tfrecord_file(directory_path, split_name)
            if tfrecord_file:
                print("Skip preprocessing for split '{}' because TFRecord file already exists at {}".format(split_name, tfrecord_file))
            else:
                target_shape = config.getImageInputShape()
                image_paths = _get_image_paths(to_split_dir(directory_path, split_name))
                preprocess_images_and_write_tfrecord(image_paths, directory_path, target_shape, split_name)
    
    if run_opts.command in ["all", "featuremaps"]:
        print("Created feature map files for splits " + str(split_names))
        for split_name in split_names:
            tfrecord_file = get_preprocessing_tfrecord_file(directory_path, split_name)
            if tfrecord_file:
                print("Start feature map generation for split '{}' with TFRecord file found at {}".format(split_name, tfrecord_file))
                create_many_feature_map_files_from_config(config, split_name)
            else:
                print("Cannot find TFRecord file for split '{}'. Please run 'preprocess' for the split and try again.")

if __name__ == '__main__':
    main()
    