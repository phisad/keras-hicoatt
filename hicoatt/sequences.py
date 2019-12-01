'''
Created on 14.03.2019

Utilities for the VQA 1.0 dataset
 
@author: Philipp
'''
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from hicoatt.dataset.answers import load_answers_by_question_from_config
from hicoatt.dataset.images.providers import ImageProvider
from hicoatt.dataset.questions import load_prepared_questions_json_from, load_questions_json_from
import numpy as np


class Vqa1Sequence(Sequence):
    """
        The sequence to generate batches of (questions, image_provider) and (answers)
        
        This holds all possible questions and answers in memory, but loads the image_provider lazily.
    """
    
    def __init__(self, questions, image_provider, vocabulary, batch_size):
        """
            @param questions: list
                The list of questions as dicts of { "question", "image_id", "question_id" }
            @param image_provider: ImageContainer
                The container to fetch the image_provider by question dicts { "question", "image_id", "question_id" }
            @param vocabulary: Vocabulary
                The vocabulary to use to encode the textual questions
            @param batch_size: int
                The batch size to use
        """
        if batch_size > 300:
            raise Exception("Batch size shouldnt be more than 300 but is " + str(batch_size))
        self.questions = questions
        self.image_provider = image_provider
        self.vocabulary = vocabulary
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.questions) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_questions = self.questions[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self._get_batch_for_input_questions(batch_questions)
    
    def _get_batch_for_input_questions(self, batch_questions):
        raise Exception("Not implemented")
    
    def one_shot_iterator(self):
        for idx in range(len(self)):
            yield self[idx]

    @staticmethod
    def create_vqa1_prediction_sequence_from_config(config, vocabulary, target_split):
        # The following are loaded as 'deep' having an own directory
        image_provider = ImageProvider.create_from_config(config, target_split)
        questions = load_questions_json_from(config.getDatasetTextDirectoryPath(), config.getPreparationCorpus(), target_split)["questions"]
        return Vqa1PredictionSequence(questions, image_provider, vocabulary, config.getBatchSize())
    
    @staticmethod
    def create_vqa1_labelled_sequence_from_config(config, vocabulary, labels, split_name):
        # The following are loaded as 'deep' having an own directory
        image_provider = ImageProvider.create_from_config(config, split_name)
        # The following are loaded as 'flat' files on the top directory
        answers = load_answers_by_question_from_config(config, split_name)
        prepared_questions = load_prepared_questions_json_from(config.getDatasetTextDirectoryPath(), split_name)
        return Vqa1LabelledInputsSequence(prepared_questions, answers, labels, image_provider, vocabulary, config.getBatchSize(), config.getNumClasses())

        
class Vqa1InputsSequence(Vqa1Sequence):
    """
        The sequence to generate batches of (questions, images)
        
        This holds all possible questions and answers in memory, but loads the images on demand.
    """
    
    def __init__(self, questions, image_provider, vocabulary, batch_size):
        super().__init__(questions, image_provider, vocabulary, batch_size)

    def _get_batch_for_input_questions(self, batch_questions):
        """
            @return: a batch of inputs as a dict of input_questions and input_images
                        where input_questions are encoded and padded textual questions
                          and input_images are the according scaled image_provider
        """
        batch_textual_questions = [entry["question"] for entry in batch_questions]
        batch_padded_questions = self.vocabulary.questions_to_encoding(batch_textual_questions)
        batch_images = self.image_provider.get_images_for_questions(batch_questions)
        return {"input_questions": batch_padded_questions, "input_images": batch_images}


class Vqa1LabelledInputsSequence(Vqa1InputsSequence):
    """
        The sequence to generate batches of (questions, image_provider) and (answers)
        
        This holds all possible questions and answers in memory, but loads the images on demand.
        
        @param answers: dict
            The dict of answers as dicts by question_id use like answers["question_id"] -> { "answer" , "answer_type" } 
        @param labels: dict
            The dictionary for a bi-directional mapping of anwser text ["idx_to_labels"] <-> class label integer ["labels_to_idx"]
            Notice: Thus the model can predict answer with multiple words without predicting the actual words.
            Moreover, this file is necessary, because we do not train on all the possible answers, but only a subset.
            Answers that are not contained in the top answers are mapped to the UNK answer class label integer at zero (0)
        @param num_classes: int
            The number of categories for the output labels
    """
    
    def __init__(self, questions, answers, labels, image_provider, vocabulary, batch_size, num_classes):
        super().__init__(questions, image_provider, vocabulary, batch_size)
        self.answers =  answers
        self.labels =  labels
        self.num_classes = num_classes 

    def __get_answer_idx(self, textual_answer):
        top_answers_by_label = self.labels["labels_to_idx"]
        if str(textual_answer) in top_answers_by_label:
            return top_answers_by_label[textual_answer]
        raise Exception("No label found for answer '{}'. Unknown answers are not allowed. Please re-prepare the dataset.".format(textual_answer))
    
    def _get_batch_for_input_questions(self, batch_questions):
        """
            @return: a batch of inputs as a dict of input_questions and input_images
                        where input_questions are encoded and padded textual questions
                          and input_images are the according scaled image_provider
                     furthermore the answers are returned as categorical one-hot encoded vocabulary integers
                        for example, the answer words integers for three questions are [37 90 1] 
                        then the categorical encoding is for answers is based on the top answer word integers
        """
        # inputs
        batch_questions_and_images = super()._get_batch_for_input_questions(batch_questions)
        
        # labels
        batch_question_ids = [entry["question_id"] for entry in batch_questions]
        batch_answers = [self.answers[str(qid)] for qid in batch_question_ids]
        batch_textual_answers = [entry["answer"] for entry in batch_answers]
        batch_encoded_answers = [self.__get_answer_idx(textual_answer) for textual_answer in batch_textual_answers]
        batch_one_hot_labels = to_categorical(batch_encoded_answers, self.num_classes)
        return (batch_questions_and_images, batch_one_hot_labels)
    

class Vqa1PredictionSequence(Vqa1InputsSequence):
    """
        The sequence to generate batches of (questions, image_provider) and returns the actual questions
        This holds all possible questions and answers in memory, but loads the images on demand.
    """
    
    def __init__(self, questions, image_provider, vocabulary, batch_size):
        super().__init__(questions, image_provider, vocabulary, batch_size)
    
    def _get_batch_for_input_questions(self, batch_questions):
        batch_questions_and_images = super()._get_batch_for_input_questions(batch_questions)
        return (batch_questions_and_images, batch_questions)
    
