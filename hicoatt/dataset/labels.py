'''
Created on 14.03.2019

@author: Philipp
'''
from collections import Counter

from hicoatt.dataset import load_json_from, store_json_to
from hicoatt.dataset.answers import load_answers_by_question_from

DEFAULT_LABELS_FILE_NAME = "vqa1_labels.json"


def load_labels_from_config(config, split_name):
    return load_labels_json_from(config.getDatasetTextDirectoryPath(), split_name)


def load_labels_json_from(directory_path_or_file, split_name=None, flat=True):
    """
        @param split_name: when given looks for the sub-directory or file in the flat directory
        @param flat: when True looks for a file in the given directory, otherwise looks into the sub-directory
        @param force: when split and flat, then force to look for the split name (only use when you know what you are doing)
    """
    lookup_filename = DEFAULT_LABELS_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        lookup_filename = "vqa1_labels_{}.json".format(split_name) 
#        print("No support for split specific labels loading. Please just name the file to use to " + lookup_filename)
        
    return load_json_from(directory_path_or_file, lookup_filename)


def _store_labels(directory_path, labels, other_labels, split_name):
    top_counts = sum([count for (_, count) in labels])
    other_counts = sum([count for (_, count) in other_labels])
    print("Top labels will cover {:3.2f}% of the answers".format(100 - (other_counts / top_counts * 100)))
    
    # Having an UNK label results in worse learning
    # labels.insert(0, ("UNK", other_counts))
    
    labels_json = {}
    labels_json["top_counts"] = top_counts
    labels_json["other_counts"] = other_counts
    labels_json["labels_to_idx"] = dict([(label, idx) for idx, (label, _) in enumerate(labels)])
    labels_json["idx_to_labels"] = dict([(idx  , label) for idx, (label, _) in enumerate(labels)])
    labels_json["labels_to_count"] = dict([(label, count) for idx, (label, count) in enumerate(labels)])
    
    lookup_filename = DEFAULT_LABELS_FILE_NAME
    if split_name:    
        lookup_filename = "vqa1_labels_{}.json".format(split_name) 
    
    return store_json_to(labels_json, directory_path, lookup_filename)


def create_labels_from_config(config, split_name):
    return create_labels(config.getDatasetTextDirectoryPath(), config.getNumClasses(), split_name)


def create_labels(directory_path, num_classes, split_name=None):
    """
        By default, the answer file is read flat and the labels file is also written flat.
        
        @param num_classes: int
            The number of top labels to determine. This always includes the unknown label (at position zero).
            When the number of labels is less than num_classes, then the returned list has the same size as labels found plus the unknown label.
            
            Thus the number of labels is num_classes. 
    """
    answers = load_answers_by_question_from(directory_path, split_name)
    textual_answers = [entry["answer"] for entry in answers.values()]
    counter = Counter(textual_answers)
    return _store_labels(directory_path, counter.most_common()[:num_classes], counter.most_common()[num_classes:], split_name)

