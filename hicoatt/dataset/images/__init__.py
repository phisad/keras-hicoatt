import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from hicoatt.dataset import determine_file_path
import numpy as np
from hicoatt import SPLIT_TRAIN, SPLIT_VALIDATE, SPLIT_TEST_DEV, SPLIT_TEST


def store_numpy_to(data, target_directory_path_or_file, lookup_file_name=None):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(target_directory_path_or_file, lookup_file_name, to_read=False)    
    
    with open(file_path, "wb") as f:
        np.save(f, data)
    return file_path


def load_numpy_from(directory_or_file, lookup_filename=None):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(directory_or_file, lookup_filename)
    
    with open(file_path, "rb") as f:
        data = np.load(f)
    return data


def get_infix_from_config(config, split_name):
    if split_name == SPLIT_TRAIN:
        return config.getTrainingImageNameInfix()
    if split_name == SPLIT_VALIDATE:
        return config.getValidationImageNameInfix()
    if split_name in [SPLIT_TEST, SPLIT_TEST_DEV]:
        return config.getTestImageNameInfix()
    raise Exception("Cannot determine image name infix for split " + split_name)

    
def _exists_image_path_by_id(image_prefix, image_id, directory_path=None, file_ending="jpg"):
    file_path = to_image_path_by_id(image_prefix, image_id, directory_path, file_ending)
    return os.path.isfile(file_path)

    
def to_image_path_by_id(image_prefix, image_id, directory_path=None, file_ending="jpg"):
    """
        Returns MSCOCO naming pattern e.g. COCO_<split_name>_<image_id>
        
        For example COCO_train2014_000000000009.jpg
    """
    file_name = "{}_{:012}.{}".format(image_prefix, image_id, file_ending)
    if directory_path:
        return "/".join([directory_path, file_name])
    return file_name


def _get_image(image_file_path, target_shape):
    with load_img(image_file_path, target_size=target_shape) as image:
        imagearr = img_to_array(image)
    return imagearr


def _get_image_paths(directory_path):
    return ["/".join([directory_path, file]) for file in os.listdir(directory_path) if file.endswith('.jpg')]


def _extract_image_id(image_path):
    """
        Returns the image id from a MSCOCO naming pattern e.g. COCO_<split_name>_<image_id>
        
        For example COCO_train2014_000000000009.jpg has the image id 9
    """
    filename = os.path.splitext(image_path)[0]
    image_id = filename.split("_")[2]
    image_id = int(image_id)
    return image_id


def _extract_to_image_ids_ordered(image_paths):
    """ Its critical to keep the order here """
    return np.array([_extract_image_id(image_path) for image_path in image_paths])

