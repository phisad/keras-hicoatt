import tensorflow as tf
import time

SPLIT_TRAIN = "train"
SPLIT_VALIDATE = "validate"
SPLIT_TEST = "test"
SPLIT_TEST_DEV = "test_dev"
SPLIT_TRAINVAL = "trainval"

def create_reduce_lr_on_plateau(callbacks_listing):
    """ Thi is propably not necessary when using adam optimizer """
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                                monitor="categorical_accuracy", 
                                factor=0.5, patience=1, verbose=1, 
                                mode="max")
    callbacks_listing.append(reduce_lr)
    print("Added lr reduction after epochs without improvement")

def create_early_stopping(callbacks_listing):
    early_stopping = tf.keras.callbacks.EarlyStopping(
                                monitor="categorical_accuracy",
                                min_delta=0.0001, patience=5, verbose=1,
                                mode="max", baseline=0.25)
    callbacks_listing.append(early_stopping)
    print("Added early stopping after 5 epochs")


def create_tensorboard_from_save_path(log_path, callbacks_listing, base_path):
    """
        Create tensorboard callback based on a given save_path. 
        The dirname from the given path is used to construct the log path.
    """
    time_tag = time.strftime("%H-%M-%S", time.gmtime())
    
    tagged_log_path = "{}/{}/{}".format(base_path, log_path, time_tag)
    print("- Tensorboard: " + tagged_log_path)
    
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tagged_log_path)
    callbacks_listing.append(tensorboard)
    return tagged_log_path

def create_csv_logger(tagged_log_path, callbacks_listing):
    logname = "/".join([tagged_log_path, "training.csv"])
    print("- CSV log: " + logname)
    csvlogger = tf.keras.callbacks.CSVLogger(logname)
    callbacks_listing.append(csvlogger)

def create_checkpointer(tagged_log_path, model_name, callbacks_listing, store_per_epoch=False):
    """
        Create checkpoint callback based on a given log_path. 
    """
    if store_per_epoch:
        model_name = model_name + ".{epoch:03d}.h5"
    else:
        model_name = model_name + ".h5"
    model_path = "/".join([tagged_log_path, model_name])
    print("- Checkpoint: " + model_path)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path, monitor="categorical_accuracy", mode="max", save_best_only=True, verbose=1)
    callbacks_listing.append(checkpointer)
    return model_path
