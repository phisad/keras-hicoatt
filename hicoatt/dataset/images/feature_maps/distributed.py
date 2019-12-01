'''
Created on 22.03.2019

    Creates many numpy files which contain a feature map array each. The file name is indicating the image id.
    
    - The image id and preprocessed images are loaded from a TFRecord file
    - The image ids are gathered from the file paths
    - The image ids and feature maps are jointly stored to a single file (expecting the order is kept)
    
    This workflow is a good option, when the data does not fit into memory.
    
@author: Philipp
'''
import tensorflow
from tensorflow.python.keras.applications.vgg19 import preprocess_input

from hicoatt.dataset.images import load_numpy_from, to_image_path_by_id, \
    store_numpy_to, get_infix_from_config
from hicoatt.dataset.images.preprocessing.tfrecords import make_dataset
from hicoatt.model.visual_embeddings import image_features_model
from hicoatt.dataset.images.preprocessing import get_tfrecord_filename
from hicoatt.scripts import OPTION_DRY_RUN
from hicoatt import SPLIT_VALIDATE, SPLIT_TRAIN, SPLIT_TEST, SPLIT_TEST_DEV


def load_feature_map_by_image_id_from_many(image_ids, directory_path, image_prefix, split_name=None):
    """
        Returns a dict of {image_id : feature_map} for each given image_id.
    """
    if split_name:
        directory_path = "/".join([directory_path, split_name])
    return dict([(image_id, load_numpy_from(to_image_path_by_id(image_prefix, image_id, directory_path, file_ending="npy"))) 
                for image_id in image_ids])

    
def __store_feature_maps_to_many(image_ids, feature_maps, directory_path, image_prefix, split_name=None):
    """
        @param ids: array 1d  (batches)
            A list of image ids in the same order as the feature maps
        @param maps: array 3d (batches, features, maps)
            An array of feature maps in the same order as the ids
    """
    if split_name:
        directory_path = "/".join([directory_path, split_name])
    return [store_numpy_to(feature_map, to_image_path_by_id(image_prefix, image_id, directory_path, file_ending="npy")) 
            for image_id, feature_map in zip(image_ids, feature_maps)]


def __get_batch_size(config, run_opts):
    if run_opts["batch_size"]:
        return run_opts["batch_size"]
    batch_size = config.getPreparationBatchSize()
    if batch_size and batch_size > 0:
        return batch_size
    return 32


def __get_expected_num_images(config, run_opts, split_name):
    if run_opts["num_images"]:
        return run_opts["num_images"]
    
    if split_name == SPLIT_TRAIN:
        expected_num_images = config.getNumberOfTrainingImages()
        if expected_num_images and expected_num_images > 0:
            return expected_num_images
        
    if split_name == SPLIT_VALIDATE:
        expected_num_images = config.getNumberOfValidationImages()
        if expected_num_images and expected_num_images > 0:
            return expected_num_images
        
    if split_name in [SPLIT_TEST, SPLIT_TEST_DEV]:
        expected_num_images = config.getNumberOfTestImages()
        if expected_num_images and expected_num_images > 0:
            return expected_num_images
        
    return None


def __is_dryrun(run_opts):
    if run_opts[OPTION_DRY_RUN]:
        return run_opts[OPTION_DRY_RUN]
    return False


def create_many_feature_map_files_from_config(config, split_name):
    """
        Reads the image files from the sub-directories given as split names.
        
        Creates the according feature map files in the top directory.
    """
    run_opts = config.run_opts
    print("Run opts: " + str(run_opts))
    
    batch_size = __get_batch_size(config, run_opts)
    expected_num_images = __get_expected_num_images(config, run_opts, split_name)
    is_dryrun = __is_dryrun(run_opts)
    
    images_top_directory = config.getDatasetImagesDirectoryPath()
    target_shape = config.getImageInputShape()
    image_features_size = config.getImageFeaturesSize()
    
    image_infix = get_infix_from_config(config, split_name)
    image_prefix = "COCO_" + image_infix
    return create_many_feature_map_files(images_top_directory, image_prefix, image_features_size, target_shape, batch_size, split_name, expected_num_images, is_dryrun)


def create_many_feature_map_files(directory_path, image_prefix,
                                  image_features_size,
                                  input_shape,
                                  batch_size=32,
                                  split_name=None,
                                  expected_num_images=None,
                                  dryrun=False):
    """
        Given a TFRecord file, creates multiple numpy files: one for each pair of image id and feature map.
    """
    tfrecord_path = "/".join([directory_path, get_tfrecord_filename(split_name)])
    dataset = make_dataset(tfrecord_path)
    dataset = dataset.batch(batch_size)
    dataset = dataset.make_one_shot_iterator()
    input_images, input_ids = dataset.get_next()
    
    processed_count = 0
        
    with tensorflow.Session() as sess:
        model = image_features_model(image_features_size, input_shape)
        try:
            while True:
                batch_images, batch_ids = sess.run([input_images, input_ids])
                batch_images = preprocess_input(batch_images, mode="caffe")
                batch_images = model.predict_on_batch(batch_images)
                __store_feature_maps_to_many(batch_ids, batch_images, directory_path, image_prefix, split_name)
                
                processed_count += len(batch_images)
                if expected_num_images:
                    print(">> Processing images {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_images, processed_count / expected_num_images * 100), end="\r")
                else:
                    print(">> Processing images {:d}".format(processed_count), end="\r")
                
                if dryrun and processed_count > 100:
                    raise Exception("Dryrun finished")
        except:
            print("Processed all images: {}".format(processed_count))
        
