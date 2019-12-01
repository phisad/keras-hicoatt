'''
Created on 07.03.2019

@author: Philipp
'''
import configparser
import os.path as path
import os
import numpy as np

import json
SECTION_DATASET = "DATASETS"
OPTION_TEXTUAL_DATASET_DIRECTORY_PATH = "TextualDatasetDirectoryPath"
OPTION_TEXTUAL_PREPARATION_CORPUS = "TextualPreparationCorpus"
OPTION_VOCABULAY_INCLUDE_ANSWERS = "VocabularyIncludeAnswers"
OPTION_IMAGE_DATASET_DIRECTORY_PATH = "ImageDatasetDirectoryPath"
OPTION_TRAINING_IMAGE_NAME_INFIX = "TrainingImageNameInfix"
OPTION_NUMBER_OF_TRAINING_IMAGES = "NumberOfTrainingImages"
OPTION_VALIDATION_IMAGE_NAME_INFIX = "ValidationImageNameInfix"
OPTION_NUMBER_OF_VALIDATION_IMAGES = "NumberOfValidationImages"
OPTION_TEST_IMAGE_NAME_INFIX = "TestImageNameInfix"
OPTION_NUMBER_OF_TEST_IMAGES = "NumberOfTestImages"
OPTION_PREPARATION_BATCH_SIZE = "PreparationBatchSize"
OPTION_PREPARATION_USING_NLTK_TOKENIZER = "PreparationUsingNltkTokenizer"

SECTION_MODEL = "MODEL"
OPTION_PRINT_MODEL_SUMMARY = "PrintModelSummary"
OPTION_MODEL_DERIVATE_NAME = "ModelDerivateName"
OPTION_QUESTION_MAXIMAL_LENGTH = "QuestionMaximalLength"
OPTION_VOCABULARY_SIZE = "VocabularySize"
OPTION_IMAGE_FEATURES_SIZE = "ImageFeaturesSize"
OPTION_IMAGE_INPUT_SHAPE = "ImageInputShape"
OPTION_IMAGE_FEATURES_SIZE = "ImageFeaturesSize"
OPTION_BYPASS_IMAGE_FEATURES_COMPUTATION = "ByPassImageFeaturesComputation"
OPTION_IMAGE_TOP_LAYER = "ImageTopLayer"
OPTION_IMAGE_TOP_LAYER_DROPOUT_RATE = "ImageTopLayerDropoutRate"
OPTION_DROPOUT_RATE = "DropoutRate"

SECTION_TRAINING = "TRAINING"
OPTION_GPU_DEVICES = "GpuDevices"
OPTION_TENSORBOARD_LOGGING_DIRECTORY = "TensorboardLoggingDirectory"
OPTION_EPOCHS = "Epochs"
OPTION_BATCH_SIZE = "BatchSize"
OPTION_MODEL_TYPE = "ModelType"
OPTION_NUM_CLASSES = "NumClasses"
OPTION_USE_MULTI_PROCESSING = "UseMultiProcessing"
OPTION_WORKERS = "Workers"
OPTION_MAX_QUEUE_SIZE = "MaxQueueSize"

FILE_NAME = "configuration.ini"


class Configuration(object):

    def __init__(self, config_path=None):
        '''
        Constructor
        '''
        self.run_opts = {}
        self.config = configparser.ConfigParser()
        if not config_path:
            config_path = Configuration.config_path()
        print("Use configuration file at: " + config_path)
        self.config.read(config_path)
        
    def __getitem__(self, idx):
        return self.run_opts[idx]
    
    def __setitem__(self, key, value):
        self.run_opts[key] = value

    def getPrintModelSummary(self):
        return self.config.getboolean(SECTION_MODEL, OPTION_PRINT_MODEL_SUMMARY)
    
    def getModelDerivateName(self):
        return self.config.get(SECTION_MODEL, OPTION_MODEL_DERIVATE_NAME)
    
    def getQuestionMaximalLength(self):
        return self.config.getint(SECTION_MODEL, OPTION_QUESTION_MAXIMAL_LENGTH)
    
    def getVocabularySize(self):
        return self.config.getint(SECTION_MODEL, OPTION_VOCABULARY_SIZE)

    def getVocabularyIncludeAnswers(self):
        return self.config.getboolean(SECTION_DATASET, OPTION_VOCABULAY_INCLUDE_ANSWERS)
    
    def getPreparationCorpus(self):
        """
            One of [open-ended, multiple-choice]
        """
        return self.config.get(SECTION_DATASET, OPTION_TEXTUAL_PREPARATION_CORPUS)
    
    def getPreparationUsingNltkTokenizer(self):
        return self.config.getboolean(SECTION_DATASET, OPTION_PREPARATION_USING_NLTK_TOKENIZER)
    
    def getDatasetTextDirectoryPath(self):
        return self.config.get(SECTION_DATASET, OPTION_TEXTUAL_DATASET_DIRECTORY_PATH)
    
    def getDatasetImagesDirectoryPath(self):
        return self.config.get(SECTION_DATASET, OPTION_IMAGE_DATASET_DIRECTORY_PATH)
    
    def getTrainingImageNameInfix(self):
        return self.config.get(SECTION_DATASET, OPTION_TRAINING_IMAGE_NAME_INFIX)
    
    def getValidationImageNameInfix(self):
        return self.config.get(SECTION_DATASET, OPTION_VALIDATION_IMAGE_NAME_INFIX)
    
    def getTestImageNameInfix(self):
        return self.config.get(SECTION_DATASET, OPTION_TEST_IMAGE_NAME_INFIX)
    
    def getNumberOfTrainingImages(self):
        return self.config.getint(SECTION_DATASET, OPTION_NUMBER_OF_TRAINING_IMAGES)
    
    def getNumberOfValidationImages(self):
        return self.config.getint(SECTION_DATASET, OPTION_NUMBER_OF_VALIDATION_IMAGES)
    
    def getNumberOfTestImages(self):
        return self.config.getint(SECTION_DATASET, OPTION_NUMBER_OF_TEST_IMAGES)
    
    def getPreparationBatchSize(self):
        return self.config.getint(SECTION_DATASET, OPTION_PREPARATION_BATCH_SIZE)
    
    def getImageInputShape(self):
        shape = self.config.get(SECTION_MODEL, OPTION_IMAGE_INPUT_SHAPE)
        shape_tuple = tuple(map(int, shape.strip('()').split(',')))
        return shape_tuple

    def getImageFeaturesSize(self):
        return self.config.getint(SECTION_MODEL, OPTION_IMAGE_FEATURES_SIZE)
    
    def getImageTopLayer(self):
        return self.config.getboolean(SECTION_MODEL, OPTION_IMAGE_TOP_LAYER)
    
    def getImageTopLayerDropoutRate(self):
        return self.config.getfloat(SECTION_MODEL, OPTION_IMAGE_TOP_LAYER_DROPOUT_RATE)
    
    def getDropoutRate(self):
        return self.config.getfloat(SECTION_MODEL, OPTION_DROPOUT_RATE)
    
    def getByPassImageFeaturesComputation(self):
        return self.config.getboolean(SECTION_MODEL, OPTION_BYPASS_IMAGE_FEATURES_COMPUTATION)
    
    def getGpuDevices(self):
        return self.config.getint(SECTION_TRAINING, OPTION_GPU_DEVICES)
    
    def getTensorboardLoggingDirectory(self):
        return self.config.get(SECTION_TRAINING, OPTION_TENSORBOARD_LOGGING_DIRECTORY)

    def getEpochs(self):
        return self.config.getint(SECTION_TRAINING, OPTION_EPOCHS)

    def getBatchSize(self):
        return self.config.getint(SECTION_TRAINING, OPTION_BATCH_SIZE)

    def getModelType(self):
        return self.config.get(SECTION_MODEL, OPTION_MODEL_TYPE)
    
    def getNumClasses(self):
        return self.config.getint(SECTION_MODEL, OPTION_NUM_CLASSES)    

    def getUseMultiProcessing(self):
        return self.config.getboolean(SECTION_TRAINING, OPTION_USE_MULTI_PROCESSING)    

    def getWorkers(self):
        return self.config.getint(SECTION_TRAINING, OPTION_WORKERS)    

    def getMaxQueueSize(self):
        return self.config.getint(SECTION_TRAINING, OPTION_MAX_QUEUE_SIZE)    
    
    def dump(self):
        print("Configuration:")
        for section in self.config.sections():
            print("[{}]".format(section))
            for key in self.config[section]:
                print("{} = {}".format(key, self.config[section][key]))
                
    @staticmethod
    def config_path():
        # Lookup file in project root or install root
        project_root = os.path.dirname(os.path.realpath(__file__))
        config_path = "/".join([project_root, FILE_NAME])
        if path.exists(config_path):
            return config_path
        print("Warn: No existing 'configuration.ini' at default location " + config_path)
        
        # Lookup file in user directory
        from pathlib import Path
        home_directory = str(Path.home())
        config_path = "/".join([home_directory, "hicoatt-" + FILE_NAME])
        if path.exists(config_path):
            return config_path
        print("Warn: No existing 'hicoatt-configuration.ini' file at user home " + config_path)
        
        raise Exception("""Please place a 'configuration.ini' in the default location 
                            or a 'hicoatt-configuration.ini' in your home directory 
                            or use the run option to specify a specific file""")

