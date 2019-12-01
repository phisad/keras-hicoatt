'''
Created on 01.03.2019

The supervised training pipeline is as follows:

Preparation:
- pre-process input data
- determine a data provider
- determine a network architecture

Training:
- initialize the network variables as a trainable model
- forward propagate data through the model
- calculate loss based on the actual data labels
- back propagate the loss and update the network variables

Stopping:
- monitor the training progress by calculating metrics e.g. accuracy on a validation set 
- determine stopping strategy e.g. all data run through once or metrics not changing anymore
- stop and save the model e.g. each iteration or at the end

@author: Philipp
'''
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import load_model

from hicoatt import create_tensorboard_from_save_path, create_checkpointer, \
    create_early_stopping, create_csv_logger, SPLIT_TRAIN, SPLIT_VALIDATE
from hicoatt.model import create_baseline_model_from_config, create_hicoatt_model_from_config
from hicoatt.scripts import OPTION_DRY_RUN, OPTION_FORCE_SAVE
from hicoatt.sequences import Vqa1Sequence
from hicoatt.dataset.results import Vqa1Results
from hicoatt.dataset.labels import load_labels_from_config
from hicoatt.dataset.vocabulary import PaddingVocabulary
from tensorflow.python.keras.optimizers import RMSprop
from hicoatt.model.custom_layers import CUSTOM_LAYER_REGISTRY


def print_model_summary(model, config):
    if config.getPrintModelSummary():
        model.summary()


def __get_sequences(config, split_name):
    if split_name:
        labels = load_labels_from_config(config, split_name)
        vocabulary = PaddingVocabulary.create_vocabulary_from_config(config, split_name)
        training_sequence = Vqa1Sequence.create_vqa1_labelled_sequence_from_config(config, vocabulary, labels, split_name)
        validation_sequence = None
    else:
        labels = load_labels_from_config(config, SPLIT_TRAIN)
        vocabulary = PaddingVocabulary.create_vocabulary_from_config(config, SPLIT_TRAIN)
        training_sequence = Vqa1Sequence.create_vqa1_labelled_sequence_from_config(config, vocabulary, labels, SPLIT_TRAIN)
        validation_sequence = Vqa1Sequence.create_vqa1_labelled_sequence_from_config(config, vocabulary, labels, SPLIT_VALIDATE)
    return training_sequence, validation_sequence


def __get_callbacks(config, split_name):
    _callbacks = []
    create_early_stopping(_callbacks)
    
    model_name = config.getModelType()
    is_top = "top" if config.getImageTopLayer() else "notop"
    split = split_name if split_name else "train"
    log_path = "{}/{}/{:.2f}/{}/{}".format(model_name, split, config.getDropoutRate(), is_top, config.getModelDerivateName())
    tagged_log_path = create_tensorboard_from_save_path(log_path, _callbacks, config.getTensorboardLoggingDirectory())
    
    if not config[OPTION_DRY_RUN] or config[OPTION_FORCE_SAVE]:
        create_checkpointer(tagged_log_path, config.getModelType(), _callbacks)
    create_csv_logger(tagged_log_path, _callbacks)
    return _callbacks


def __getOptimizer(config):
    return RMSprop(lr=0.0004, rho=0.99, decay=0.00000001)


def __get_model(config, path_to_model, initial_epoch):
    """
        Loads the model from the given path or creates a new model based on the configuration.
        The model is compiled before return.
    """
    if path_to_model:
        if not initial_epoch:
            raise Exception("You have to set the initial_epoch, when continuing training")
        print("Try to load model from path: " + path_to_model)
        model = load_model(path_to_model, compile=False, custom_objects=CUSTOM_LAYER_REGISTRY)
    elif config.getModelType() == "baseline":
        print("Create baseline model")
        model = create_baseline_model_from_config(config)
    else:
        print("Create HiCoAtt model")
        model = create_hicoatt_model_from_config(config)
    _optimizer = __getOptimizer(config)
    print("Compile model with {} optimizer, categorical loss and metrics".format(str(_optimizer)))
    model.compile(optimizer=_optimizer, loss=categorical_crossentropy, metrics=[categorical_accuracy])
    print_model_summary(model, config)
    return model


def start_training(config, path_to_model=None, initial_epoch=None, split_name=None):
    training_sequence, validation_sequence = __get_sequences(config, split_name)
    model = __get_model(config, path_to_model, initial_epoch)
    _callbacks = __get_callbacks(config, split_name)
    print("Start training now")
    model.fit_generator(training_sequence,
                        validation_data=validation_sequence,
                        validation_steps=None if not config[OPTION_DRY_RUN] else 10,
                        epochs=config.getEpochs() if not config[OPTION_DRY_RUN] else 1,
                        verbose=2 if not config[OPTION_DRY_RUN] else 1,
                        steps_per_epoch=None if not config[OPTION_DRY_RUN] else 10,
                        callbacks=_callbacks,
                        use_multiprocessing=config.getUseMultiProcessing(),
                        workers=config.getWorkers(),
                        max_queue_size=config.getMaxQueueSize(),
                        initial_epoch=0 if not initial_epoch else initial_epoch)


def __is_dryrun(run_opts):
    if run_opts[OPTION_DRY_RUN]:
        return run_opts[OPTION_DRY_RUN]
    return False


import tensorflow as tf


def start_prediction(config, path_to_model, source_split, target_split):
    results = Vqa1Results.create(config, source_split)
    with tf.Session():
        # The following are loaded as 'flat' files on the top directory
        vocabulary = PaddingVocabulary.create_vocabulary_from_config(config, source_split)
        prediction_sequence = Vqa1Sequence.create_vqa1_prediction_sequence_from_config(config, vocabulary, target_split)
        
        model = __get_model(config, path_to_model, 1)
        print_model_summary(model, config)
        
        dryrun = __is_dryrun(config.run_opts)
        processed_count = 0
        expected_num_batches = len(prediction_sequence)
        try:
            for batch_inputs, batch_questions in prediction_sequence.one_shot_iterator():
                batch_predictions = model.predict_on_batch(batch_inputs)
                results.add_batch(batch_questions, batch_predictions)
                processed_count = processed_count + 1
                print(">> Processing batches {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_batches, processed_count / expected_num_batches * 100), end="\r")
                if dryrun and processed_count > 10:
                    raise Exception("Dryrun finished")
        except:
            print("Processed all images: {}".format(processed_count))
        
    results.write_vqa_results_file(path_to_model, target_split)
    results.write_human_results_file(path_to_model)
        
