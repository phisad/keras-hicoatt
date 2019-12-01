'''
Created on 14.03.2019

@author: Philipp
'''
import json

from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from hicoatt.dataset import load_json_from, store_json_to, determine_file_path
from hicoatt.dataset.questions import load_prepared_questions_json_from
from hicoatt.dataset.answers import load_answers_by_question_from
from nltk.tokenize import word_tokenize


DEFAULT_VOCABULARY_FILE_NAME = "vqa1_vocabulary.json"


def _store_tokenizer(tokenizer, target_directory_path_or_file, split_name):
    lookup_filename = DEFAULT_VOCABULARY_FILE_NAME
    if split_name:    
        lookup_filename = "vqa1_vocabulary_{}.json".format(split_name) 
    return store_json_to(json.loads(tokenizer.to_json()), target_directory_path_or_file, lookup_filename)


def get_vocabulary_file_path(config, split_name=None, flat=True):
    lookup_filename = DEFAULT_VOCABULARY_FILE_NAME
    if split_name and not flat:
        raise Exception("Only flat vocabulary path supported for now")
    if split_name and flat:
        lookup_filename = "vqa1_vocabulary_{}.json".format(split_name)
        #print("No support for split specific vocabulary loading. Please just name the file to use to " + lookup_filename)
    try:
        return determine_file_path(config.getDatasetTextDirectoryPath(), lookup_filename, to_read=True)
    except Exception:
        print("No vocabulary file found with name " + lookup_filename)
        return None



def nltk_tokenize(question):
    return word_tokenize(str(question).lower())

def create_vocabulary_file_from_config(config, split_name):
    """
        Create the vocabulary based on the split.
        Expects the answers to be in the top directory.
        Puts the vocabulary in the top directory.
    """
    directory_path = config.getDatasetTextDirectoryPath()
    questions = load_prepared_questions_json_from(directory_path, split_name)
    if questions and not isinstance(questions, list):
        raise Exception("Questions must be a listing of question dicts")
    textual_questions = [question["question"] for question in questions]
    
    if config.getPreparationUsingNltkTokenizer():
        total_count = len(textual_questions)
        processed_count = 0
        tokenized_questions = []
        for question in textual_questions:
            processed_count += 1
            print('>> Tokenizing questions %d/%d' % (processed_count, total_count), end="\r")
            tokenized_questions.append(nltk_tokenize(question))
        textual_questions = tokenized_questions
    
    tokenizer = Tokenizer(oov_token="UNK")
    print("Fit vocabulary on {} questions".format(len(questions)))
    tokenizer.fit_on_texts(textual_questions)
    
    if config.getVocabularyIncludeAnswers():
        # In the original paper implementation answers are not part of the vocabulary
        answers = load_answers_by_question_from(directory_path, split_name)
        if answers and not isinstance(answers, dict):
            raise Exception("Answers must be a dict of answer dicts")
        
        textual_answers = [answer["answer"] for answer in answers.values()]
        
        if config.getPreparationUsingNltkTokenizer():
            total_count = len(textual_answers)
            processed_count = 0
            tokenized_answers = []
            for answer in textual_answers:
                processed_count += 1
                print('>> Tokenizing answers %d/%d' % (processed_count, total_count), end="\r")
                tokenized_answers.append(nltk_tokenize(answer))
            textual_answers = tokenized_answers
            
        print("Fit vocabulary on {} answers".format(len(answers)))
        tokenizer.fit_on_texts(textual_answers)
    
    return _store_tokenizer(tokenizer, directory_path, split_name)


def load_vocabulary_file_from(directory_path_or_file, split_name=None, flat=True):
    """
        @param split_name: when given looks for the sub-directory or file in the flat directory
        @param flat: when True looks for a file in the given directory, otherwise looks into the sub-directory 
    """
    lookup_filename = DEFAULT_VOCABULARY_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:
        lookup_filename = "vqa1_vocabulary_{}.json".format(split_name) 
        #print("No support for split specific vocabulary loading. Please just name the file to use to " + lookup_filename)
        
    tokenizer_config = load_json_from(directory_path_or_file, lookup_filename)
    tokenizer = tokenizer_from_json(json.dumps(tokenizer_config))
    return tokenizer


class Vocabulary():
    
    def __init__(self, tokenizer, use_nltk):
        self.tokenizer = tokenizer
        self.use_nltk = use_nltk

    def questions_to_encoding(self, questions):
        """
            Encode the textual questions to an integer encoding. Unknown words are replaced with UNK.
            The integer max value is the vocabulary size.
            
            @param questions: the textual questions as list 
            @return: the questions as integer sequences
        """
        if self.use_nltk:
            questions = [nltk_tokenize(q) for q in questions]
        return self.tokenizer.texts_to_sequences(questions)
    
    def maximal_question_length_from_config(self, config, split_name):
        return self.maximal_question_length_from(config.getDatasetTextDirectoryPath(), split_name)
        
    def maximal_question_length_from(self, directory_path, split_name):
        """
            Determine the maximal question length based on this vocabulary.
            The questions are tokenized before counting.
            
            @param source_directory_path_or_file: the file or directory containing the textual questions json
            @return: the maximal question length
        """
        questions = load_prepared_questions_json_from(directory_path, split_name)
        textual_questions = [question["question"] for question in questions]
        return self.maximal_question_length(textual_questions)

    def maximal_question_length(self, questions):
        """
            Determine the maximal question length based on this vocabulary.
            The questions are tokenized before counting.
            
            @param questions: the textual questions as list
            @return: the maximal question length
        """
        if type(questions) is not list:
            return 0
        return max([len(question) for question in self.questions_to_encoding(questions)])
    
    def __len__(self):
        return len(self.tokenizer.word_index)
    
    @staticmethod
    def create_vocabulary_from_vocabulary_json(source_directory_path_or_file, use_nltk):
        return Vocabulary(load_vocabulary_file_from(source_directory_path_or_file), use_nltk)


class PaddingVocabulary(Vocabulary):
    """
        The padding vocabulary additional zero pads the questions to maximal length when performing an encoding.
    """
    
    def __init__(self, tokenizer, use_nltk, questions_max_length):
        """
            @param questions_max_length: the (globally) maximal length of a question
        """
        super().__init__(tokenizer, use_nltk)
        self.questions_max_length = questions_max_length

    def questions_to_encoding(self, questions):
        """
            Encode the textual questions to an integer encoding. Unknown words are replaced with UNK.
            The integer max value is the vocabulary size.
            
            The padding is attached at the end of each question up to the maximal question length.
            
            @param questions: the textual questions as list 
            @return: the padded and encoded questions
        """
        encoded_questions = super().questions_to_encoding(questions)
        padded_questions = pad_sequences(encoded_questions, maxlen=self.questions_max_length, padding="post")
        return padded_questions
    
    @staticmethod
    def create_vocabulary_from_config(config, split_name=None):
        return PaddingVocabulary.create_vocabulary_from_vocabulary_json(config.getDatasetTextDirectoryPath(), 
                                                                        config.getPreparationUsingNltkTokenizer(), 
                                                                        config.getQuestionMaximalLength(), 
                                                                        split_name)
    
    @staticmethod
    def create_vocabulary_from_vocabulary_json(source_directory_path_or_file, use_nltk, questions_max_length, split_name=None):
        return PaddingVocabulary(load_vocabulary_file_from(source_directory_path_or_file, split_name), use_nltk, questions_max_length)
