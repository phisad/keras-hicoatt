'''
Created on 03.04.2019

@author: Philipp
'''
from hicoatt import SPLIT_TRAINVAL, SPLIT_TRAIN, SPLIT_VALIDATE
from hicoatt.dataset import store_json_to, load_json_from
from hicoatt.dataset.answers import load_answers_by_question_from
from hicoatt.dataset.labels import load_labels_json_from

DEFAULT_PREPARED_QUESTIONS_FILE_NAME = "vqa1_questions.json"

DEFAULT_PREPARED_QUESTIONS_SPLIT_FILE_NAME_PATTERN = "vqa1_questions_{}.json"

DEFAULT_QUESTION_FILE_NAME_PATTERN = "v1_{}_mscoco_questions.json"


def load_questions_json_from(directory_path_or_file, corpus_type, split_name=None, flat=False):
    """
        @param split_name: when given looks for the sub-directory or file in the flat directory
        @param flat: when True looks for a file in the given directory, otherwise looks into the sub-directory 
    """
    lookup_filename = DEFAULT_QUESTION_FILE_NAME_PATTERN.format(corpus_type)
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        raise Exception("Not supported to have source question files on the same level as the top dataset directory")
        
    return load_json_from(directory_path_or_file, lookup_filename)


def load_prepared_questions_json_from_config(config, split_name):
    return load_prepared_questions_json_from(config.getDatasetTextDirectoryPath(), split_name)


def load_prepared_questions_json_from(directory_path_or_file, split_name=None, flat=True):
    lookup_filename = DEFAULT_PREPARED_QUESTIONS_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        lookup_filename = DEFAULT_PREPARED_QUESTIONS_SPLIT_FILE_NAME_PATTERN.format(split_name) 
        
    return load_json_from(directory_path_or_file, lookup_filename)

    
def __store_prepared_questions_as_file(prepared_questions, target_directory_path_or_file, split_name=None):
    lookup_filename = DEFAULT_PREPARED_QUESTIONS_FILE_NAME
    if split_name:    
        lookup_filename = DEFAULT_PREPARED_QUESTIONS_SPLIT_FILE_NAME_PATTERN.format(split_name) 
    return store_json_to(prepared_questions, target_directory_path_or_file, lookup_filename)


def create_prepared_questions_file_from_config(config, split_names):
    if split_names and not isinstance(split_names, list):
        split_names = [split_names]
    return [__create_prepared_questions_file_from_config(config, split_name) for split_name in split_names]


def __create_prepared_questions_file_from_config(config, split_name):
    """ 
        For the normal train split we only filter the training questions
        to simulate other questions in the validate split
        for train+val split we have to filter both splits.
        
        In both cases we create a single questions.json file
    """
    directory_path = config.getDatasetTextDirectoryPath()
    corpus_type = config.getPreparationCorpus()
    if split_name == SPLIT_TRAINVAL:
        labels_json = load_labels_json_from(directory_path, SPLIT_TRAINVAL)
        training_questions_json = load_questions_json_from(directory_path, corpus_type, SPLIT_TRAIN)
        valdiation_questions_json = load_questions_json_from(directory_path, corpus_type, SPLIT_VALIDATE)
        questions = training_questions_json["questions"] + valdiation_questions_json["questions"]
    else:
        # use training labels also for the validate split to filter the questions
        labels_json = load_labels_json_from(directory_path, SPLIT_TRAIN)
        questions_json = load_questions_json_from(directory_path, corpus_type, split_name)
        questions = questions_json["questions"]
        
    answers_json = load_answers_by_question_from(directory_path, split_name)
    prepared_questions = [question for question in questions
                          if answers_json[str(question["question_id"])]["answer"] 
                          in labels_json["labels_to_idx"]]
    total_amount = len(questions)
    reduced_amount = len(prepared_questions)
    print("Reduced questions from {} to {} ({:.2f}%)".format(total_amount, reduced_amount, reduced_amount / total_amount * 100))        
    return __store_prepared_questions_as_file(prepared_questions, directory_path, split_name)
