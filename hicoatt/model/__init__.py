"""
The hiCoAtt perform training as following (https://github.com/jiasenlu/HieCoAttenVQA/blob/master/train.lua)

(a) The three parts of the network are handled in parallel.
 
    Notice: The colon is for implementing methods that pass self as the first parameter. 
            So x:bar(3,4)should be the same as x.bar(x,3,4).
    
  protos.word:training()
  protos.phrase:training()
  protos.ques:training()
  protos.recursive_atten:training()
      
(b) Then their results are combined in the recursive answer predictor

  local word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward(
      {data.questions, data.images}))
  
  local conv_feat, p_ques, p_img = unpack(protos.phrase:forward(
      {word_feat, data.ques_len, img_feat, mask}))
      
  local q_ques, q_img = unpack(protos.ques:forward(
      {conv_feat, data.ques_len, img_feat, mask}))
      
  local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img}
  local out_feat = protos.recursive_atten:forward(feature_ensemble)
  
  -- forward the language model criterion
  local loss = protos.crit:forward(out_feat, data.answer)
"""
from hicoatt.model.attention import coattention_affinity_graph, recursive_attention_graph
from hicoatt.model.textual_embeddings import word_embeddings_graph, phrase_embeddings_graph, question_embeddings_graph
from hicoatt.model.visual_embeddings import image_features_graph_flatten, image_features_graph_with_gap, \
    bypass_image_features_graph_with_gap, \
    bypass_image_features_graph_flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dot, Concatenate, LSTM, Flatten, GlobalAveragePooling2D, Reshape, Input
from hicoatt.model.custom_layers import BinaryMaskedSoftmax, BinaryMasking

def create_baseline_model_from_config(config):
    print("Dropout rate: {:.2f}".format(config.getDropoutRate()))
    if config.getByPassImageFeaturesComputation():
        image_features_graph = bypass_image_features_graph_with_gap(config.getImageFeaturesSize(), config.getImageTopLayer(), config.getImageTopLayerDropoutRate())
        print("Bypass image feature generation")
    else:
        image_features_graph = image_features_graph_with_gap(config.getImageInputShape())
    return __create_baseline_model(config.getQuestionMaximalLength(), config.getVocabularySize(), image_features_graph, config.getNumClasses(), config.getDropoutRate())


def __create_baseline_model(questions_max_length, vocabulary_size, image_features_graph, num_classes, dropout_rate, embedding_size=512):
    """
        Create a baseline model which takes the same inputs and has the same outputs as the HiCoAtt model.
        
        @param vocalbulary_size: the (globally) amount of known words
        @param questions_max_length: the (globally) maximal length of a question
        
        @return: the model
                    with input_images of shape (448, 448, 3) and
                    with input_questions of length T (see textual_embeddings.preprocessing for preparing the questions)
                    which outputs (B x C), C = one-hot encoded num_classes
    """
    vgraph, vinputs = image_features_graph
        
    wgraph, winputs = word_embeddings_graph(questions_max_length, vocabulary_size, dropout_rate, embedding_size)
    wgraph = LSTM(embedding_size, name="baseline_lstm")(wgraph)
    
    ograph = Concatenate()([vgraph, wgraph])
    ograph = Dense(num_classes, activation="softmax", name="baseline_predictor")(ograph)
    
    print("Visual input shape: " + str(vinputs.shape))
    print("Textual input shape: " + str(winputs.shape))
    model = Model(inputs=[winputs, vinputs], outputs=ograph)
    return model


def create_hicoatt_model_from_config(config):
    print("Dropout rate: {:.2f}".format(config.getDropoutRate()))
    if config.getImageTopLayer():
        print("Image Top Dropout rate: {:.2f}".format(config.getImageTopLayerDropoutRate()))
    if config.getByPassImageFeaturesComputation():
        image_features_graph = bypass_image_features_graph_flatten(config.getImageFeaturesSize(), config.getImageTopLayer(), config.getImageTopLayerDropoutRate())
        print("Bypass image feature generation")
    else:
        image_features_graph = image_features_graph_flatten(config.getImageFeaturesSize(), config.getImageInputShape())
    return create_hicoatt_model(config.getQuestionMaximalLength(), config.getVocabularySize(), image_features_graph, config.getNumClasses(), config.getDropoutRate())


def create_hicoatt_model(questions_max_length, vocabulary_size, image_features_graph, num_classes, dropout_rate, embedding_size=512):
    """
        Create the HiCoAtt model based on the question max length and vocabulary size.
        
        The model in the original paper is trained with embedding size 512 on the top answers.
        
        @param vocalbulary_size: the (globally) amount of known words
        @param questions_max_length: the (globally) maximal length of a question
        
        @return: the model
                    with input_images of shape (448, 448, 3) and
                    with input_questions of length T (see textual_embeddings.preprocessing for preparing the questions)
                    which outputs (B x C), C = one-hot encoded num_classes
    """
    vgraph, vinputs = image_features_graph
        
    wgraph, winputs, wmask = word_embeddings_graph(questions_max_length, vocabulary_size, dropout_rate, embedding_size)
    wqfeat, wvfeat = coattention_affinity_graph("word", wgraph, vgraph, dropout_rate, embedding_size, wmask)
    
    pgraph = phrase_embeddings_graph(wgraph, dropout_rate, embedding_size)
    pqfeat, pvfeat = coattention_affinity_graph("phrase", pgraph, vgraph, dropout_rate, embedding_size, wmask)
    
    qgraph = question_embeddings_graph(pgraph, dropout_rate, embedding_size)
    qqfeat, qvfeat = coattention_affinity_graph("question", qgraph, vgraph, dropout_rate, embedding_size, wmask)
    
    rgraph = recursive_attention_graph(wqfeat, wvfeat, pqfeat, pvfeat, qqfeat, qvfeat, num_classes, dropout_rate, embedding_size)
    
    print("Visual input shape: " + str(vinputs.shape))
    print("Textual input shape: " + str(winputs.shape))
    model = Model(inputs=[winputs, vinputs], outputs=rgraph)
    return model
