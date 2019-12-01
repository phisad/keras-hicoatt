'''
Created on 21.03.2019

@author: Philipp
'''
from io import BytesIO
import sys

from multiprocessing import Pool
import tqdm
from tensorflow.keras.preprocessing.image import load_img
from hicoatt.dataset.images import _extract_image_id
from hicoatt.dataset.images.preprocessing.tfrecords import write_tfrecord
from hicoatt.dataset.images.preprocessing import get_tfrecord_filename
from hicoatt.dataset import determine_file_path


def get_preprocessing_tfrecord_file(target_directory_path_or_file, split_name=None):
    try:
        return determine_file_path(target_directory_path_or_file, get_tfrecord_filename(split_name), to_read=True)
    except Exception:
        return None


def preprocess_images_and_write_tfrecord(image_paths, target_directory_path_or_file, target_shape, split_name=None):
    split_dict_listing = __load_and_preprocess_data_into_parallel(__paths_to_dicts(image_paths, target_shape), number_of_processes=20)  
    
    for imaget in split_dict_listing:
        if imaget[0] == "Failure":
            print(imaget)  

    def collect_success(dicts):
        return [imaget[1] for imaget in dicts if imaget[0] == "Success"]

    split_dicts = collect_success(split_dict_listing)  
    
    tf_record_name = get_tfrecord_filename(split_name)
    
    return write_tfrecord(split_dicts, target_directory_path_or_file, tf_record_name)


def __paths_to_dicts(file_paths, target_shape):
    dicts = []
    for file_path in file_paths:
        d = {}
        d["path"] = file_path
        d["width"] = target_shape[0]
        d["height"] = target_shape[1]
        dicts.append(d)
    return dicts


def __load_and_preprocess_data_into_parallel(dicts, number_of_processes):
    results = []
    with Pool(processes=number_of_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(__load_and_preprocess_single_defended, dicts), total=len(dicts)):
            results.append(result)
    return results

        
def __load_and_preprocess_single_defended(imaged):
    try:
        __load_and_preprocess_single(imaged)
        return ("Success", imaged)
    except:
        err_msg = sys.exc_info()[0]
        err = sys.exc_info()[1]
        error = (imaged["path"], err_msg, err)
        return ("Failure", error)


def __load_and_preprocess_single(imaged):
    """ Shrink images """
    with load_img(imaged["path"], target_size=(imaged["width"], imaged["height"])) as image:
        with BytesIO() as raw:
            image.save(raw, "JPEG")
            imaged["data"] = raw.getvalue()
        imaged["imageid"] = _extract_image_id(imaged["path"])
