'''
Created on 20.03.2019

@author: Philipp
'''
from hicoatt.dataset import load_annotations_json_from, load_json_from, \
    store_json_to
from hicoatt import SPLIT_TRAINVAL, SPLIT_VALIDATE, SPLIT_TRAIN
    
DEFAULT_ANSWERS_FILE_NAME = "vqa1_answers.json"


def __load_answers_by_question_from_annotations(directory_path, split_name):
    annotations_json = load_annotations_json_from(directory_path, split_name, flat=False)
    answers_by_question_id = _to_answers_by_question_ids(annotations_json)
    return answers_by_question_id


def _to_answers_by_question_ids(annotations_json):
    answers_by_question = {}
    for annotation in annotations_json["annotations"]:
        question_id = annotation["question_id"]
        question_answer = annotation["multiple_choice_answer"]
        question_answer_type = annotation["answer_type"]
        answers_by_question[question_id] = { "answer" : question_answer, "answer_type" : question_answer_type }
    return answers_by_question


def create_answers_by_question_file_from_config(config, split_names):
    """
        Reads the annotation files from the sub-directories given as split names.
        
        Creates the according answers files in the top directory.
    """
    anwser_files = [__create_answers_by_question_file(config.getDatasetTextDirectoryPath(), split_name) for split_name in split_names]
    return anwser_files 


def __create_answers_by_question_file(directory_path, split_name=None):
    """
        Read the files deep and stores them flat.
        
        Read the annotations file and create a map of 
            { question_id : { "answer_type",  "answer" } }  
        where "answer_type" is on of ["other", "yes/no", "number"]
        and   "answer" is the ground-truth answer (majority over 10 answers)
        
        The map is stored as a .json file in the target directory.
    """
    if split_name == SPLIT_TRAINVAL:
        training_answers_by_question_id = __load_answers_by_question_from_annotations(directory_path, SPLIT_TRAIN)
        validation_answers_by_question_id = __load_answers_by_question_from_annotations(directory_path, SPLIT_VALIDATE)
        answers_by_question_id = {**training_answers_by_question_id, **validation_answers_by_question_id}
    else:
        answers_by_question_id = __load_answers_by_question_from_annotations(directory_path, split_name)
    return _store_answers_by_question_as_file(answers_by_question_id, directory_path, split_name)


def _store_answers_by_question_as_file(labels_map, target_directory_path_or_file, split_name=None):
    lookup_filename = DEFAULT_ANSWERS_FILE_NAME
    if split_name:
        lookup_filename = "vqa1_answers_{}.json".format(split_name) 
    return store_json_to(labels_map, target_directory_path_or_file, lookup_filename)


def load_answers_by_question_from_config(config, split_name):
    """ We expect that the file is located in the top directory """
    directory_path = config.getDatasetTextDirectoryPath()
    return load_answers_by_question_from(directory_path, split_name)


def load_answers_by_question_from(directory_path_or_file, split_name=None, flat=True):
    """
        @param split_name: when given looks for the sub-directory or file in the flat directory
        @param flat: when True looks for a file in the given directory, otherwise looks into the sub-directory 
    """
    lookup_filename = DEFAULT_ANSWERS_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        lookup_filename = "vqa1_answers_{}.json".format(split_name) 
        
    return load_json_from(directory_path_or_file, lookup_filename)

