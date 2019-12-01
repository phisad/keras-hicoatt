'''
Created on 21.03.2019

@author: Philipp
'''
import tensorflow as tf
import numpy as np
import sys


def tfrecord_inputs(file_path, batch_size=100):
    dataset = make_dataset(file_path)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    sample_op = iterator.get_next()
    return sample_op


def load_tfrecord_in_memory(tfrecord_file, directory):
    tfrecord_file_path = "/".join([directory, tfrecord_file])
    load_tfrecord_in_memory_from_path(tfrecord_file_path)

    
def load_tfrecord_in_memory_from_path(tfrecord_file_path):
    tf.reset_default_graph()
    inputs = tfrecord_inputs(tfrecord_file_path)
    images_all = []
    imageids_all = []
    with tf.Session() as sess:
        try:
            while True:
                images, imageids = sess.run(inputs)
                images_all.extend(images)
                imageids_all.extend(imageids)
        except:
            print("Loaded all inputs: {}".format(len(images_all)))
    return np.array(images_all), np.array(imageids_all)


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(data, height, width, imageid):
    """
        Creates a TFRecord example image entry.
    """
    return tf.train.Example(features=tf.train.Features(feature={
      'image/data': bytes_feature(data),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/imageid': int64_feature(imageid)
  }))


def write_tfrecord(image_tuples, target_directory, filename):
    """
        Creates a TFRecord file from a map of image files paths to class ids
    Args:
        image_tuples: List of image tuples 
        target_directory: where to put the file
        filename: how to name the file
    """
    tf.reset_default_graph()
    
    # image_path = tf.placeholder(dtype=tf.string)
    # image_raw = tf.read_file(image_path)
        
    errors = []
    num_images = len(image_tuples)
    target_file_path = "/".join([target_directory, filename])
    with tf.Session() as sess:
        with tf.python_io.TFRecordWriter(target_file_path) as tfrecord_writer:
            processed_count = 0
            for image_tuple in image_tuples:
                # Show progress
                processed_count += 1
                print('>> Converting image %d/%d' % (processed_count, num_images), end="\r")
                
                try:
                    example = image_to_tfexample(
                        image_tuple["data"],
                        image_tuple["height"],
                        image_tuple["width"],
                        image_tuple["imageid"]
                    )
                    tfrecord_writer.write(example.SerializeToString())
                except:
                    err_msg = sys.exc_info()[0]
                    err = sys.exc_info()[1]
                    errors.append((image_tuple["path"], err_msg, err))
                
    print()            
    print("Errors: {}".format(len(errors)))
    if len(errors) > 0:
        for (error_file, info, err) in errors:
            print("Error one file: {} because: {} / {}".format(error_file, info , err))
    return target_file_path


def read_sample(example_raw):
    """
        Read a single TFRecord example and converting into an image
    Args:
        The TFRecord example that represents an image
    """
    example = tf.parse_single_example(
        example_raw,
        features={
            'image/data': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/imageid': tf.FixedLenFeature([], tf.int64),
        })
    
    image_height = tf.cast(example['image/height'], tf.int32)
    image_width = tf.cast(example['image/width'], tf.int32)
    image = tf.image.decode_image(example['image/data'], channels=3)
    image = tf.reshape(image, (image_height, image_width, 3))
    
    imageid = example["image/imageid"]
    return image, imageid


def make_dataset(tfrecord_filepath):
    """
        Returns a dataset ready to process a TFRecord file
    """
    dataset = tf.data.TFRecordDataset(tfrecord_filepath)
    dataset = dataset.map(read_sample)
    return dataset 

