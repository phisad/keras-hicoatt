'''
Created on 09.08.2019

@author: Philipp
'''
from hicoatt.dataset.questions import load_prepared_questions_json_from
from hicoatt.dataset.answers import load_answers_by_question_from_config
from hicoatt.dataset.images import get_infix_from_config,\
    _exists_image_path_by_id, to_image_path_by_id, store_numpy_to
from hicoatt.model import create_hicoatt_model_from_config
from tensorflow.python.keras.models import Model
from hicoatt.dataset.vocabulary import PaddingVocabulary
from hicoatt.dataset.images.providers import ImageProvider
from hicoatt.dataset import store_json_to
import numpy as np

def start_extraction(config, path_to_model, source_split_name, target_split_name):
    create_attention_maps_from_config(config, path_to_model, source_split_name, target_split_name)

def create_attention_maps_from_config(config, path_to_model, source_split_name, target_split_name):
    """ load the questions """
    """ { "image_id": 487025, "question": "Is there a shadow?","question_id": 4870251 } """
    prepared_questions = load_prepared_questions_json_from(config.getDatasetTextDirectoryPath(), source_split_name)
    
    """ filter the questions by other """
    """ "90": { "answer": "pink and yellow",  "answer_type": "other" } """
    gt_answers = load_answers_by_question_from_config(config, source_split_name)
    other_questions = [q for q in prepared_questions if gt_answers[str(q["question_id"])]["answer_type"] == "other"]
    print("Other questions", len(other_questions))# == 138460
    
    """ prepare the questions as per image """
    import collections
    questions_by_image = collections.defaultdict(list)
    [questions_by_image[q["image_id"]].append(q) for q in other_questions]
    print("Images", len(questions_by_image.keys()))
    #assert len(questions_by_image.keys()) == 92874
    
    """ filter the image by validation images """
    split_dir = config.getDatasetImagesDirectoryPath() + "/" + target_split_name
    infix = get_infix_from_config(config, target_split_name)
    image_prefix = "COCO_{}".format(infix)
    target_images = [questions_by_image[image_id] for # get the questions as list
        image_id in questions_by_image.keys() if 
        _exists_image_path_by_id(image_prefix, image_id, split_dir)  # when image is in target split
        ]
    # a list of lists then
    print("Target Images", len(target_images))
    #assert len(target_images) == 30684
    
    # load the model
    model = create_hicoatt_model_from_config(config)
    model.load_weights(path_to_model, by_name=True)
    
    # attach to the attention layers
    watt = model.get_layer("word_image_attention")
    patt = model.get_layer("phrase_image_attention")
    qatt = model.get_layer("question_image_attention")
    matt = Model(inputs=model.inputs, outputs=[watt.output, 
            patt.output, 
            qatt.output])
    
    """ for each image produce attention maps """
    vocabulary = PaddingVocabulary.create_vocabulary_from_config(config, source_split_name)
    image_provider = ImageProvider.create_single_split_provider_from_config(config, target_split_name)
    
    processed_count = 0
    expected_num_batches = len(target_images)
    for image_questions in target_images:
        processed_count = processed_count + 1
        print(">> Processing images {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_batches, processed_count / expected_num_batches * 100), end="\r")
        image_id = image_questions[0]["image_id"] #image id is the same for all questions
        batch_questions = image_questions
        batch_textual_questions = [entry["question"] for entry in batch_questions]
        batch_padded_questions = vocabulary.questions_to_encoding(batch_textual_questions)
        batch_images = image_provider.get_images_for_questions(batch_questions) # this returns n times the same but prepared image
        
        """ store the attention as numpy """
        attention_maps = np.array(matt.predict({"input_images":batch_images, "input_questions":batch_padded_questions}))
        attention_map_path = to_image_path_by_id(image_prefix, image_id, split_dir, file_ending="bqx")
        store_numpy_to(attention_maps, attention_map_path)
        
        """ store the question infos as json"""
        quids = [q["question_id"] for q in image_questions]
        answers = [gt_answers[str(quid)] for quid in quids]
        question_info = {"questions":image_questions, "answers":answers}
        attention_info_path = to_image_path_by_id(image_prefix, image_id, split_dir, file_ending="iqx")
        store_json_to(question_info, attention_info_path)