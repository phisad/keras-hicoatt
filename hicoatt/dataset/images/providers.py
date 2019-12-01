'''
Created on 22.03.2019

@author: Philipp
'''

from tensorflow.keras.applications.vgg19 import preprocess_input
from hicoatt.dataset.images import to_image_path_by_id, _get_image, load_numpy_from, \
    get_infix_from_config, _exists_image_path_by_id
import numpy as np
from hicoatt import SPLIT_VALIDATE, SPLIT_TRAINVAL, SPLIT_TRAIN, SPLIT_TEST_DEV,\
    SPLIT_TEST


class ImageProvider():
    """
        Given a batch of question ids, the provider returns a batches of corresponding images.
    """

    def __init__(self, directory_path=None, prefix="COCO_train2014"):
        """
            @param directory_path: str
                Directory containing the image_provider with names of COCO_train2014_<12-digit-image-id> for example COCO_train2014_000000000009
            @param prefix: str
                The image file name prefixed to the image id part
        """
        self.directory_path = directory_path
        self.prefix = prefix
    
    def get_images_for_questions(self, questions):
        image_ids = [entry["image_id"] for entry in questions]
        return self.get_images_for_image_ids(image_ids)

    def get_images_for_image_ids(self, image_ids):
        raise Exception("Not implemented")
    
    def _get_image_for_image_id(self, image_id):
        raise Exception("Not implemented")
    
    def _has_image_for_image_id(self, image_id):
        raise Exception("Not implemented")
    
    @staticmethod
    def create_multi_split_provider_from_config(config, split_names):
        providers = [ImageProvider.create_single_split_provider_from_config(config, split_name) for split_name in split_names]
        return Vqa1MultipleExclusiveProvider(providers)

    @staticmethod
    def create_single_split_provider_from_config(config, split_name):
        directory_path = "/".join([config.getDatasetImagesDirectoryPath(), split_name])
        image_infix = get_infix_from_config(config, split_name)
        if config.getByPassImageFeaturesComputation():
            return Vqa1FileSystemFeatureMapProvider(directory_path, prefix="COCO_" + image_infix)
        else:
            return Vqa1FileSystemImageProvider(directory_path, prefix="COCO_" + image_infix, vgg_like=True, target_size=config.getImageInputShape())    
        
    @staticmethod
    def create_from_config(config, split_name):
        if split_name == SPLIT_TRAINVAL:
            print("Identified trainval split for image provider")
            return ImageProvider.create_multi_split_provider_from_config(config, [SPLIT_TRAIN, SPLIT_VALIDATE])
        elif split_name == SPLIT_TEST_DEV:
            return ImageProvider.create_single_split_provider_from_config(config, SPLIT_TEST)
        else:
            return ImageProvider.create_single_split_provider_from_config(config, split_name)


class Vqa1MultipleExclusiveProvider(ImageProvider):
    """
        This provider is useful, when there are multiple existing directories for images. Then this provider manages the retrieval from these.
        
        Given a batch of question ids, the provider reads batches of feature maps from a directory on the file system.
        
        Here, the given providers are asked if they can succeed the request. Then the first accepting provider will be used to complete the request.
        
        Notice: This might be slow during training, when the file system is the bottleneck. 
    """

    def __init__(self, providers):
        super().__init__(None, None)
        self.providers = providers
        
    def _get_image_for_image_id(self, image_id):
        for provider in self.providers:
            if provider._has_image_for_image_id(image_id):
                return provider._get_image_for_image_id(image_id)
        raise Exception("Cannot serve image for image id '{}'".format(image_id))
    
    def get_images_for_image_ids(self, image_ids):
        """
            @return: the images in the same order as the image ids
        """
        images = np.array([self._get_image_for_image_id(image_id) for image_id in image_ids]) 
        return images

    
class Vqa1FileSystemFeatureMapProvider(ImageProvider):
    """
        Given a batch of question ids, the provider reads batches of feature maps from a directory on the file system.
        
        Notice: This might be slow during training, when the file system is the bottleneck. 
    """

    def __init__(self, directory_path=None, prefix="COCO_train2014"):
        super().__init__(directory_path, prefix)
    
    def get_images_for_image_ids(self, image_ids):
        """
            @param questions: the list of questions as dicts of { "question", "image_id", "question_id" }
            
            @return: the image in the same order as questions
        """
        feature_map_file_paths = [to_image_path_by_id(self.prefix, image_id, file_ending="npy") for image_id in image_ids]
        feature_maps = np.array([load_numpy_from(self.directory_path, image_file_path) for image_file_path in feature_map_file_paths])
        return feature_maps
    
    def _get_image_for_image_id(self, image_id):
        """ Overwritten to determine the file ending """
        image_file_path = to_image_path_by_id(self.prefix, image_id, file_ending="npy")
        return load_numpy_from(self.directory_path, image_file_path) 
    
    def _has_image_for_image_id(self, image_id):
        """ Overwritten to determine the file ending """
        return _exists_image_path_by_id(self.prefix, image_id, self.directory_path, file_ending="npy")
        
        
class Vqa1FileSystemImageProvider(ImageProvider):
    """
        Given a batch of question ids, the provider reads batches of images from a directory on the file system.
        
        The images are preprocessed on the fly like in image preparation:
        - reduce image size to target size
        - apply vgg like preprocessing if applicable
        
        Notice: This might be slow during training, when the file system is the bottleneck. 
    """
    
    def __init__(self, directory_path=None, prefix="COCO_train2014", vgg_like=True, target_size=(448, 448)):
        """
            @param vgg_like: bool
                When true, then the image_provider are prepared for VGG use according to the 'mode'. 
                
                mode: One of "caffe", "tf" or "torch".
                    - caffe: will convert the image_provider from RGB to BGR,
                        then will zero-center each color channel with
                        respect to the ImageNet dataset,
                        without scaling.
                    - tf: will scale pixels between -1 and 1, sample-wise.
                    - torch: will scale pixels between 0 and 1 and then
                        will normalize each channel with respect to the
                        ImageNet dataset.
                        
                We use mode 'tf' because thats the backend here.
        """
        super().__init__(directory_path, prefix)
        self.vgg_like = vgg_like
        if target_size and len(target_size) == 3:
            target_size = (target_size[0], target_size[1])
        self.target_size = target_size
        
    def get_images_for_image_ids(self, image_ids):
        """
            @param questions: the list of questions as dicts of { "question", "image_id", "question_id" }
            
            @return: the image in the same order as questions
        """
        image_file_paths = [to_image_path_by_id(self.prefix, image_id, self.directory_path) for image_id in image_ids]
        images = np.array([_get_image(image_file_path, self.target_size) for image_file_path in image_file_paths])
        if self.vgg_like:
            images = preprocess_input(images, mode="caffe")
        return images

    def _get_image_for_image_id(self, image_id):
        """ Overwritten to determine the file ending """
        image_file_path = to_image_path_by_id(self.prefix, image_id, self.directory_path)
        return np.array(_get_image(image_file_path, self.target_size)) 
    
    def _has_image_for_image_id(self, image_id):
        """ Overwritten to determine the file ending """
        return _exists_image_path_by_id(self.prefix, image_id, self.directory_path)
    