'''
Created on 20.03.2019

    Creates a single numpy file which contains a structured array of { image_id : feature_map } pairs.
    
    - The image file paths are located and loaded by a Keras sequence
    - The feature maps are calculated using the image model sequence-wise
    - The image ids are gathered from the file paths
    - The image ids and feature maps are jointly stored to a single file (expecting the order is kept)
    
    This is a shortcut, when everything can be loaded into memory.

@author: Philipp
'''
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import Sequence

from hicoatt.dataset.images import _get_image_paths, _extract_to_image_ids_ordered, \
    store_numpy_to, load_numpy_from, _get_image
from hicoatt.model.visual_embeddings import image_features_model
import numpy as np
from hicoatt.scripts import OPTION_DRY_RUN

DEFAULT_FEATURES_FILE_NAME = "vqa1_feature_maps.npy"


def __store_feature_maps_by_image_id_to_single(image_ids, feature_maps, target_directory_path_or_file, split_name):
    """
        @param ids: array 1d  (batches)
            A list of image ids in the same order as the feature maps
        @param maps: array 3d (batches, features, maps)
            An array of feature maps in the same order as the ids
    """
    lookup_file_name = DEFAULT_FEATURES_FILE_NAME
    if split_name:
        lookup_file_name = "vqa1_feature_maps_{}.npy".format(split_name)
    
    data = np.zeros(len(image_ids), dtype=[("id", "i4"), ("features", "f4", (feature_maps.shape[1], feature_maps.shape[2]))])
    data["id"] = image_ids
    data["features"] = feature_maps
     
    return store_numpy_to(data, target_directory_path_or_file, lookup_file_name)


def load_feature_maps_by_image_id_from_single(directory_path_or_file, split_name=None, flat=True):
    lookup_filename = DEFAULT_FEATURES_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        lookup_filename = "vqa1_feature_maps_{}.json".format(split_name) 
        
    data = load_numpy_from(directory_path_or_file, lookup_filename)

    image_ids = data["id"]
    feature_maps = data["features"]
    return dict(zip(image_ids, feature_maps))


class ImageSequence(Sequence):
    
    def __init__(self, file_paths, target_size, batch_size=32, vgg_like=True):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.target_size = target_size
        self.vgg_like = vgg_like
    
    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_file_paths = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = np.stack([_get_image(image_file_path, self.target_size) for image_file_path in batch_file_paths])
        if self.vgg_like:
            batch_images = preprocess_input(batch_images, mode="caffe")
        return batch_images
    

def create_single_feature_maps_file_from_config(config, split_names):
    """
        Reads the image files from the sub-directories given as split names.
        
        Creates the according feature map files in the top directory.
    """
    images_top_directory = config.getDatasetImagesDirectoryPath()
    
    feature_map_files = []
    for split_name in split_names:
        directory_path = "/".join([images_top_directory, split_name])
        image_paths = _get_image_paths(directory_path)
        
        target_shape = config.getImageInputShape()
        image_features_size = config.getImageFeaturesSize()
        feature_map_file = create_single_feature_maps_file(image_paths, images_top_directory, image_features_size, target_shape, split_name, config.run_opts)
        feature_map_files.append(feature_map_file)
    return feature_map_files 


def create_single_feature_maps_file(image_paths, target_directory_path_or_file, image_features_size, target_shape, split_name=None, run_opts={}):
    """
        Create a single big numpy based feature map file containing all image id and feature map pairs.
    """
    print(run_opts)
    model = image_features_model(image_features_size, target_shape)
    sequence = ImageSequence(image_paths, target_shape, 32 if not "batch_size" in run_opts else run_opts["batch_size"])
    
    _steps = None
    if OPTION_DRY_RUN in run_opts and run_opts[OPTION_DRY_RUN]:
        _steps = 10
        
    image_ids = _extract_to_image_ids_ordered(image_paths)
    feature_maps = model.predict_generator(sequence, verbose=1,
                                           steps=_steps,
                                           use_multiprocessing=False if not "use_multi" in run_opts else run_opts["use_multi"],
                                           workers=1 if not "workers" in run_opts else run_opts["workers"],
                                           max_queue_size=10 if not "queues" in run_opts else run_opts["queues"])
    if OPTION_DRY_RUN in run_opts and run_opts[OPTION_DRY_RUN]:
        image_ids = image_ids[:feature_maps.shape[0]] 
    return __store_feature_maps_by_image_id_to_single(image_ids, feature_maps, target_directory_path_or_file, split_name)
    
