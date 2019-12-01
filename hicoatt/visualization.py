'''
Created on 05.04.2019

@author: Philipp
'''
from hicoatt.model.attention import coattention_affinity_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from hicoatt.sequences import Vqa1InputsSequence
from hicoatt.dataset.images.providers import Vqa1FileSystemImageProvider, \
    Vqa1FileSystemFeatureMapProvider
from hicoatt.dataset.vocabulary import PaddingVocabulary

__NO_DROPOUT = 0


def show_many_with_alpha(images, alphas, max_rows, max_cols, figsize=(20, 20), titles=None, titles_hspace=.2, plot_title=None):
    plt.figure(figsize=figsize, dpi=300)

    for idx, (image, alpha) in enumerate(zip(images, alphas)):
        row = idx // max_cols
        col = idx % max_cols

        ax = plt.subplot2grid((max_rows, max_cols), (row, col))
        if titles != None:
            subtitle_conf = {"fontsize": 2}
            ax.set_title(titles[idx], fontdict=subtitle_conf, loc="left", pad=2)
        ax.axis("off")
        ax.imshow(alpha, cmap='gray', aspect="auto")
        ax.imshow(image, aspect="auto", alpha=0.3)
        
    if titles == None:
        plt.subplots_adjust(wspace=.05, hspace=.05)        
    else:
        plt.subplots_adjust(wspace=.05, hspace=titles_hspace)
        
    if plot_title:
        plt.suptitle(plot_title)
        
    plt.show()

    
def __get_model(path_to_model):
    if path_to_model:
        print("Try to load model from path: " + path_to_model)
        model = load_model(path_to_model)
    return model


def upscale_attention(attention, target_shape):
    attention_shape = np.shape(attention)
    assert len(attention_shape) == 2 
    sqr_shape = np.sqrt(attention_shape[1]).astype("uint8")
    attention = np.reshape(attention, (-1, sqr_shape, sqr_shape, 1))
    if len(target_shape) == 3:
        target_shape = (target_shape[0], target_shape[1])
    attention = [array_to_img(a).resize(target_shape) for a in attention]
    attention = np.array([img_to_array(a) for a in attention])
    attention = attention.astype("uint8")
    attention = np.squeeze(attention)
    return attention


def visualize_image_attention_with_config(path_to_model, level_name, samples, image_rows, image_cols, config):
    
    image_directory = "/".join([config.getDatasetImagesDirectoryPath(), "test"])
    
    provider = Vqa1FileSystemFeatureMapProvider(image_directory, prefix="COCO_test2015")
    vocabulary = PaddingVocabulary.create_vocabulary_from_config(config, "trainval")
    sequence = Vqa1InputsSequence(samples, provider, vocabulary, config.getBatchSize())
    
    model = load_model(path_to_model)
    model = Model(inputs=model.input, outputs=model.get_layer(level_name + "_image_attention").output)
    
    target_shape = config.getImageInputShape()
    attention = model.predict_generator(sequence)
    attention = upscale_attention(attention, target_shape)
    
    provider = Vqa1FileSystemImageProvider(image_directory, prefix="COCO_test2015", vgg_like=False, target_size=target_shape)
    sample_image_ids = [sample["image_id"] for sample in samples]
    sample_images = provider.get_images_for_image_ids(sample_image_ids)
    sample_images = sample_images.astype('uint8')
    
    sample_answers = [sample["answer"] for sample in samples] 
    sample_questions = [sample["question"] for sample in samples] 
    titles = ["Question: {} \n Answer: {}".format(question, answer) for question, answer in zip(sample_questions, sample_answers) ]
    show_many_with_alpha(sample_images, attention, image_rows, image_cols, figsize=(4, 3), titles=titles)


def visualize_coattention_with_features(path_to_model, level_name,
                        question_features, image_features,
                        images, titles,
                        image_rows, image_cols,
                        target_shape):
    """ Notice: Its rather unlikely to have the features already. This is more a testing showcase. """
    
    image_features_shape = np.shape(image_features)
    image_feature_size = image_features_shape[1]
    
    question_features_shape = np.shape(question_features)
    question_feature_size = question_features_shape[1]
     
    model = coattention_affinity_model(level_name, question_feature_size, image_feature_size, __NO_DROPOUT, 512)
    model.load_weights(path_to_model, by_name=True)
    model = Model(inputs=model.input, outputs=model.get_layer(level_name + "_image_attention").output)
    
    attention = model.predict({"image_features":image_features, "question_features": question_features})
    attention = upscale_attention(attention, target_shape)
    
    show_many_with_alpha(images, attention, image_rows, image_cols, figsize=(4, 3), titles=titles)
