
DEFAULT_TFRECORD_FILE_NAME = "vqa1_images.tfrecord"


def get_tfrecord_filename(split_name=None):
    if split_name:
        return "vqa1_images_{}.tfrecord".format(split_name)
    return DEFAULT_TFRECORD_FILE_NAME
