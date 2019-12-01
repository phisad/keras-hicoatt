'''
Created on 31.03.2019

@author: Philipp
'''

import numpy as np
import os
from hicoatt.dataset import store_json_to
from hicoatt.dataset.labels import load_labels_from_config

import zipfile
from hicoatt import SPLIT_TEST_DEV, SPLIT_TEST

    
def store_zip_file(result_file_path, directory_path, filename):
    print("Creating results.zip")
    with zipfile.ZipFile("/".join([directory_path, filename]), mode='w') as zf:
        zf.write(result_file_path, compress_type=zipfile.ZIP_DEFLATED)


class Vqa1Results():
    
    def __init__(self, labels):
        """ list questions-prediction tuples """
        self.results = []
        self.labels = labels
        
    def add_batch(self, batch_questions, batch_predictions):
        prediction_classes = np.argmax(batch_predictions, axis=1)
        zipped = list(zip(batch_questions, prediction_classes))
        self.results.extend(zipped)

    def write_vqa_results_file(self, path_to_model, target_split):
        """
            Results Format
            results = [result]
            
            result{
            "question_id": int,
            "answer": str
            }
        """
        try:
            directory_path = os.path.dirname(path_to_model)
            results = [{"question_id" : question["question_id"],
                        "answer" : self.labels["idx_to_labels"][str(prediction_idx)]
                       } 
                       for (question, prediction_idx) in self.results]
            target_split_name = target_split
            if target_split == SPLIT_TEST_DEV:
                target_split_name = "test-dev2015"
            if target_split == SPLIT_TEST:
                target_split_name = "test2015"
            result_file_path = store_json_to(results, directory_path, "vqa_OpenEnded_mscoco_{}_hicoatt_results.json".format(target_split_name))
        except Exception as e:
            print("Cannot write vqa results file: " + str(e))
            
        try:
            store_zip_file(result_file_path, directory_path, "results.zip")
        except Exception as e:
            print("Cannot write results zip file: " + str(e))
            
    def write_human_results_file(self, path_to_model):
        """
            Results Format
            results = [result]
            
            result{
            "question": int,
            "answer": str,
            "image_id": int,
            }
        """
        try:
            directory_path = os.path.dirname(path_to_model)
            results = [{"question" : question["question"],
                        "image_id" : question["image_id"],
                        "answer" : self.labels["idx_to_labels"][str(prediction_idx)]
                       } 
                       for (question, prediction_idx) in self.results]
            store_json_to(results, directory_path, "vqa_human_results.json")
        except Exception as e:
            print("Cannot write human results file: " + str(e))

    @staticmethod
    def create(config, split_name):
        labels = load_labels_from_config(config, split_name)
        return Vqa1Results(labels)
