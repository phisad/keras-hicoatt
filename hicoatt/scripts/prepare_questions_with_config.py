#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser

from hicoatt import SPLIT_VALIDATE, SPLIT_TRAIN
from hicoatt.configuration import Configuration
from hicoatt.dataset.answers import create_answers_by_question_file_from_config
from hicoatt.dataset.labels import create_labels_from_config
from hicoatt.dataset.questions import create_prepared_questions_file_from_config
from hicoatt.dataset.vocabulary import Vocabulary, get_vocabulary_file_path, create_vocabulary_file_from_config


def __get_split_or_default(run_opts, default_split):
    split_name = default_split
    if run_opts.split_name:
        split_name = run_opts.split_name
    return split_name


def __get_splits_or_default(run_opts, default_splits):
    split_names = default_splits
    if run_opts.split_name:
        split_names = [run_opts.split_name]
    return split_names


def main():
    parser = ArgumentParser("Prepare the VQA 1.0 dataset for training")
    parser.add_argument("command", help="""One of [answers, labels, questions, vocabulary, all]. 
                        1.Step| answers:  Create all question answer pairs.
                        2.Step| labels: Create the class labels by choosing the configured amount of top answers. Requires to create the answers file before.
                        3.Step| questions: Reduce the questions to those that contain the top answer labels. Requires to create the labels file before.
                        4.Step| vocabulary: Create the vocabulary. Also determines vocabulary size and questions maximal length. Requires to create the questions file before. 
                        all: All of the above""")
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    parser.add_argument("-s", "--split_name", help="""The split name to prepare the training data for. One of [train, validate, test, trainval]. 
                        The split name determines sub-directory to lookup in the dataset directories.
                        A special split name is 'trainval' which combines both train and validate splits.
                        Notice: When the split name is not specified for training then the standard split of train and validate will be prepared. 
                        This is usually the wanted behavior. It is not possible for now to combine e.g. train and test.
                        """)
    
    run_opts = parser.parse_args()
    
    if run_opts.configuration:
        config = Configuration(run_opts.configuration)
    else:
        config = Configuration()
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.getGpuDevices())
    
    print("Starting preparation: {}".format(run_opts.command))
    
    if run_opts.command in ["all", "answers"]:
        print("\n1. Step: Determine the overall answers from the large annotation files")
        split_names = __get_splits_or_default(run_opts, [SPLIT_TRAIN, SPLIT_VALIDATE])
        answers_file_paths = create_answers_by_question_file_from_config(config, split_names)
        print("Created answer files at " + str(answers_file_paths))
        
    if run_opts.command in ["all", "labels"]:
        print("\n2. Step: Determine the top answers and store them as labels")
        split_name = __get_split_or_default(run_opts, SPLIT_TRAIN)
        labels_file_path = create_labels_from_config(config, split_name)
        print("Created top labels file at " + labels_file_path)
    
    if run_opts.command in ["all", "questions"]:
        print("\n3. Step: Filter the original questions file to those that result in top answers (to avoid unknown labels)")
        split_names = __get_splits_or_default(run_opts, [SPLIT_TRAIN, SPLIT_VALIDATE])
        questions_file_paths = create_prepared_questions_file_from_config(config, split_names)
        print("Created filtered question file at " + str(questions_file_paths))
        
    if run_opts.command in ["all", "vocabulary"]:
        print("\n4. Step: Create the vocabulary from the filtered questions and determine the questions max length")
        split_name = __get_split_or_default(run_opts, SPLIT_TRAIN)
        vocabulary_file_path = get_vocabulary_file_path(config, split_name) 
        if vocabulary_file_path:
            print("Found vocabulary file at " + vocabulary_file_path)
        else:
            print("Creating vocabulary file for split " + split_name)
            vocabulary_file_path = create_vocabulary_file_from_config(config, split_name)
        
        vocab = Vocabulary.create_vocabulary_from_vocabulary_json(vocabulary_file_path, config.getPreparationUsingNltkTokenizer())
        vocab_size = len(vocab)
        print("Determined vocabulary size " + str(vocab_size))
        
        split_names = __get_splits_or_default(run_opts, [SPLIT_TRAIN, SPLIT_VALIDATE])
        for split_name in split_names:    
            maximal = vocab.maximal_question_length_from_config(config, split_name)
            print("Determined maximal question length in {} split: {} ".format(split_name, str(maximal)))

        
if __name__ == '__main__':
    main()
    
