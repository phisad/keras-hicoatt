import os
import json

DEFAULT_ANNOTATION_FILE_NAME = "v1_mscoco_annotations.json"

def load_annotations_json_from(directory_path_or_file, split_name=None, flat=False):
    """
        @param split_name: when given looks for the sub-directory or file in the flat directory
        @param flat: when True looks for a file in the given directory, otherwise looks into the sub-directory 
    """
    lookup_filename = DEFAULT_ANNOTATION_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        raise Exception("Not yet supported to have file on the same level as the directory")
    
    return load_json_from(directory_path_or_file, lookup_filename)


def determine_file_path(directory_or_file, lookup_filename, to_read=True):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = directory_or_file
    if os.path.isdir(directory_or_file):
        if lookup_filename == None:
            raise Exception("Cannot determine source file in directory without lookup_filename")
        file_path = "/".join([directory_or_file, lookup_filename])
    if to_read and not os.path.isfile(file_path):
        raise Exception("There is no such file in the directory to read: " + file_path)
    return file_path


def load_json_from(directory_or_file, lookup_filename=None):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(directory_or_file, lookup_filename)
    print("Loading JSON from " + file_path)
    with open(file_path) as json_file:
        json_content = json.load(json_file)
    return json_content


def store_json_to(json_content, directory_or_file, lookup_filename=None):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(directory_or_file, lookup_filename, to_read=False)
    #print("Persisting JSON to " + file_path)    
    with open(file_path, "w") as json_file:
        json.dump(json_content, json_file, indent=4, sort_keys=True)
        
    return file_path


def to_split_dir(directory_path, split_name):
    return "/".join([directory_path, split_name])
