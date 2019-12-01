'''
Created on 04.03.2019

@author: Philipp
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Add, Dropout, Dot, Input, Concatenate, Flatten, TimeDistributed
from hicoatt.model.custom_layers import BinaryMaskedSoftmax

def coattention_affinity_graph(name, question_features, image_features, dropout_rate, embedding_size=512, question_mask=None):
    """
        @param name: the name of the context in which this graph is used e.g. word, phrase or question level
        @param question_features: textual features for the co-attention
        @param image_features: visual features for the co-attention
        @param embedding_size: size of the feature dimension for the hidden layers
    """
    agraph = affinity_graph(name, question_features, image_features, embedding_size)
    cgraph = coattention_graph(name, question_features, image_features, agraph, dropout_rate, embedding_size, question_mask)
    return cgraph


def coattention_affinity_model(model_name, question_max_length, image_features_size, dropout_rate, embedding_size=512):
    """
        Wrapper model for the coattention affinity graph e.g. for testing
        
        @param question_max_length: the (globally) maximal length of a question
        @param image_features_size: length of the flatten visual feature maps e.g. 196 for 14 x 14 maps
        @param embedding_size: size of the feature dimension for the hidden layers
        
        @return: the coattention model with outputs [(B x H), (B x H)], H = embedding size
    """
    image_features = Input(shape=(image_features_size, embedding_size), name="image_features")
    question_features = Input(shape=(question_max_length, embedding_size), name="question_features")
    cagraph = coattention_affinity_graph(model_name, question_features, image_features, dropout_rate)
    return Model(inputs=[question_features, image_features], outputs=cagraph)
    

def affinity_graph(name, question_features, image_features, embedding_size=512):
    """
        Compute affinity matrix by combining features Q (T x d) and V (N x d) to C (T x N).
        Therefore we introduce the weight matrix W (d x d) to compute Q (T x d) x W (d x d) x V_T (d x N)  
        
        question_features_shape: T x d, T = question max-length   , d = embedding size     = 512
        image_features_shape   : N x d, N = feature-map size = 196, d = feature-map amount = 512
    
        @param name: the name of the context in which this graph is used e.g. word, phrase or question level
        @param question_features: textual features for the co-attention
        @param image_features: visual features for the co-attention
        @param image_features_size: length of the flatten visual feature maps e.g. 196 for 14 x 14 maps
        
        @return: the affinity graph with tensor T x N
    """
    # we combine the right-hand side V (N x d) x W (d x d) = R (N x d) 
    agraph = Dense(embedding_size, name=name + "_affinity_image_features_embedding")(image_features)
    # we combine the left-hand side at axis=2 Q (T x d) x R (N x d) = C (T x N)
    agraph = Dot(axes=(2, 2), name=name + "_affinity")([question_features, agraph])
    agraph = Activation(activation="tanh", name=name + "_affinity_activation")(agraph)
    return agraph


def affinity_model(question_max_length, image_features_size, embedding_size=512):
    """
        Wrapper model for the affinity graph e.g. for testing
        
        @param question_max_length: the (globally) maximal length of a question
        @param image_features_size: length of the flatten visual feature maps e.g. 196 for 14 x 14 maps
        @param embedding_size: the embedding size, both question and image features have to share
                               the embedding size for the question features is the hidden layer size
                               the embedding size for the images features is the amount of feature maps
        
        @return: the affinity model with output T x N
    """
    image_features = Input(shape=(image_features_size, embedding_size), name="image_features")
    question_features = Input(shape=(question_max_length, embedding_size), name="question_features")
    affinity_matrix = affinity_graph("graph", question_features, image_features, embedding_size)
    return Model(inputs=[question_features, image_features], outputs=affinity_matrix, name="affinity_model")

def coattention_graph(name, question_features, image_features, affinity_matrix, dropout_rate, embedding_size=512, question_mask=None):
    """
        @param name: the name of the context in which this graph is used e.g. word, phrase or question level
        @param question_features: textual features for the co-attention
        @param image_features: visual features for the co-attention
        @param affinity_matrix: the correlation matrix for both question and image features
        @param embedding_size: size of the feature dimension for the hidden layers
    """
    # we compute V (N x d) x W (d x k) = V_w (N x k) (we set k to embedding size here, as in the paper)
    image_features_weighted = Dense(embedding_size, name=name + "_coatt_image_features_embedding")(image_features)
    # we compute Q (T x d) x W (d x k) = Q_w (T x k) (we set k to embedding size here, as in the paper)
    question_features_weighted = Dense(embedding_size, name=name + "_coatt_question_features_embedding")(question_features)
    
    # we combine C (T x N) x V_w (N x k) x  = V_a (T x k)
    qagraph = Dot(axes=(2, 1), name=name + "_coatt_image_features_transform")([affinity_matrix, image_features_weighted])
    # we combine Q_w (T x k) + V_a (T x k) = H_q (k x T)
    qagraph = Add(name=name + "_coatt_question_sum")([question_features_weighted, qagraph])
    qagraph = Activation("tanh", name=name + "_coatt_question_activation")(qagraph)
    qagraph = Dropout(rate=dropout_rate, name=name + "_coatt_question_dropout")(qagraph)
    # we apply w (1 x k) on each column k of H_q (T x k)
    # therefore T is the time-dimension, so we get a_q (T x 1)
    qagraph = TimeDistributed(Dense(1, name=name + "_question_attention_kernel"), name=name + "_question_attention_embedding")(qagraph)
    qagraph = Flatten(name=name + "_question_attention_flatten")(qagraph)
    """ Maybe apply mask here, otherwise the softmax applies prob.mass on padded values """
    if question_mask != None:
        print("Apply masking for {} level question attention".format(name))
        qagraph = BinaryMaskedSoftmax(name=name + "_question_attention")([qagraph, question_mask])
    else:
        qagraph = Activation("softmax", name=name + "_question_attention")(qagraph)
    # we combine Q (T x d) x a_q (T x 1) = A_coatt (d x 1) (one question attention feature vector per sample of embedding size)
    qagraph = Dot(axes=(1, 1), name=name + "_question_attention_feature")([question_features, qagraph])
    
    # we combine C (T x N) x Q_w (T x k)) x  = Q_a (N x k)
    vagraph = Dot(axes=(1, 1), name=name + "_coatt_question_features_transform")([affinity_matrix, question_features_weighted])
    # we combine V_w (N x k) + Q_a (N x k) = H_v (N x k)
    vagraph = Add(name=name + "_coatt_image_sum")([image_features_weighted, vagraph])
    vagraph = Activation("tanh", name=name + "_coatt_image_activation")(vagraph)
    vagraph = Dropout(rate=dropout_rate, name=name + "_coatt_image_dropout")(vagraph)
    # we apply w (1 x k) on each column k of H_v (N x k)
    # therefore N is the time-dimension, so we get a_v (N x 1)
    vagraph = TimeDistributed(Dense(1, name=name + "_image_attention_kernel"), name=name + "_image_attention_embedding")(vagraph)
    vagraph = Flatten(name=name + "_image_attention_flatten")(vagraph)
    vagraph = Activation("softmax", name=name + "_image_attention")(vagraph)
    # we combine V (N x d) x a_v (N x 1) = A_coatt (d x 1)
    vagraph = Dot(axes=(1, 1), name=name + "_image_attention_feature")([image_features, vagraph])
    
    return qagraph, vagraph


def coattention_model(question_features_size, image_features_size, dropout_rate, embedding_size=512):
    """
        Wrapper model for the coattention graph e.g. for testing
        
        @param question_features_size: maximal question length to be supported
        @param image_features_size: length of the flatten visual feature maps e.g. 196 for 14 x 14 maps
        @param embedding_size: size of the feature dimension for the hidden layers
        
        @return: the partial coattention model (without affinity) with outputs [(B x H), (B x H)], H = embedding size
    """
    image_features = Input(shape=(image_features_size, embedding_size), name="image_features")
    question_features = Input(shape=(question_features_size, embedding_size), name="question_features")
    affinity_matrix = Input(shape=(question_features_size, image_features_size), name="affinity_matrix")
    [question_coattention_features, image_coattention_features] = coattention_graph("graph", question_features, image_features, affinity_matrix, dropout_rate, embedding_size)
    return Model(inputs=[question_features, image_features, affinity_matrix], outputs=[question_coattention_features, image_coattention_features])


def recursive_attention_graph(word_coatt_features, word_coatt_visual_features,
                               phrase_coatt_features, phrase_coatt_visual_features,
                               question_coatt_features, question_coatt_visual_features,
                               num_classes, dropout_rate, 
                               embedding_size=512, question_embedding_size=1024):
    """
        @param word_coatt_features: word level attention features for the question
        @param word_coatt_visual_features: word level attention features for the image
        @param phrase_coatt_features: phrase level attention features for the question
        @param phrase_coatt_visual_features: phrase level attention features for the image
        @param question_coatt_features: question level attention features for the question
        @param question_coatt_visual_features: question level attention features for the image
        @param embedding_size: size of the feature dimension for the hidden layers
        @param num_classes: the amount of num_classes to be predictable
    """
    print("Question embedding size: " + str(question_embedding_size))
    
    word_level_features = Add(name="recursive_word_add")([word_coatt_features, word_coatt_visual_features])
    word_level_features = Dropout(rate=dropout_rate, name="recursive_word_dropout")(word_level_features)
    rgraph = Dense(embedding_size, activation="tanh", name="recursive_word_level")(word_level_features)
    
    phrase_level_features = Add(name="recursive_phrase_add")([phrase_coatt_features, phrase_coatt_visual_features])
    phrase_level_features = Concatenate(name="recursive_phrase_concatenate")([phrase_level_features, rgraph])
    phrase_level_features = Dropout(rate=dropout_rate, name="recursive_phrase_dropout")(phrase_level_features)
    rgraph = Dense(embedding_size, activation="tanh", name="recursive_phrase_level")(phrase_level_features)
    
    question_level_features = Add(name="recursive_question_add")([question_coatt_features, question_coatt_visual_features])
    question_level_features = Concatenate(name="recursive_question_concatenate")([question_level_features, rgraph])
    question_level_features = Dropout(rate=dropout_rate, name="recursive_question_dropout")(question_level_features)
    rgraph = Dense(question_embedding_size, activation="tanh", name="recursive_question_level")(question_level_features)
    
    rgraph = Dropout(rate=dropout_rate, name="recursive_prediction_dropout")(rgraph)
    rgraph = Dense(num_classes, activation="softmax", name="recursive_prediction_level")(rgraph)
     
    return rgraph


def recursive_attention_model(num_classes, dropout_rate, embedding_size=512):
    """
        Wrapper model for the recursive attention graph e.g. for testing
        
        @param embedding_size: size of the feature dimension for the hidden layers
        @param num_classes: the amount of num_classes to be predictable
        
        @return: the model with output (B x C), C = number of num_classes
    """
    wqinput, wvinput = Input(shape=(embedding_size,)), Input(shape=(embedding_size,))
    pqinput, pvinput = Input(shape=(embedding_size,)), Input(shape=(embedding_size,))
    qqinput, qvinput = Input(shape=(embedding_size,)), Input(shape=(embedding_size,))
    
    rgraph = recursive_attention_graph(wqinput, wvinput, pqinput, pvinput, qqinput, qvinput, num_classes, dropout_rate, embedding_size)
    
    return Model(inputs=[wqinput, wvinput, pqinput, pvinput, qqinput, qvinput], outputs=rgraph)

