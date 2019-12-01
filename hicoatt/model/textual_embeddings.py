'''
Created on 13.03.2019

The word-level network (https://github.com/jiasenlu/HieCoAttenVQA/blob/master/misc/word_level.lua)
The phrase-level network (https://github.com/jiasenlu/HieCoAttenVQA/blob/master/misc/phrase_level.lua)
The question-level network (https://github.com/jiasenlu/HieCoAttenVQA/blob/master/misc/ques_level.lua)

@author: Philipp
'''

from tensorflow.keras.layers import Embedding, Activation, Dropout, Input, Conv1D, Maximum, LSTM
from tensorflow.keras.models import Sequential, Model
from hicoatt.model.custom_layers import BinaryMasking


def word_embeddings_model(questions_max_length, vocabulary_size, dropout_rate, hidden_size=512):
    """
    Create the word-level language embedding model using an embedding matrix.
    
    The model should be fed with the encoded questions from preprocessing.
        
    @param questions_max_length: the (globally) maximal length of a question
    @param vocalbulary_size: the (globally) amount of known words
    @param hidden_size: the number of embedding dimensions to model 
    @return: the word embedding model 
    """
    model = Sequential()
    model.add(Embedding(vocabulary_size, hidden_size, mask_zero=True, input_shape=(questions_max_length,), name="word_embeddings"))
    model.add(Activation(activation="tanh", name="word_activation"))
    model.add(Dropout(rate=dropout_rate, name="word_dropout"))
    return model

    
def word_embeddings_graph(questions_max_length, vocabulary_size, dropout_rate, hidden_size=512):
    """
    Create the word-level language embedding model using an embedding matrix.
    
    The model should be fed with the encoded questions from preprocessing.
        
    @param questions_max_length: the (globally) maximal length of a question
    @param vocalbulary_size: the (globally) amount of known words
    @param hidden_size: the number of embedding dimensions to model 
    @return: the word embedding model 
    """
    qinputs = Input(shape=(questions_max_length,), name="input_questions")
    qmask = BinaryMasking(name="word_masking")(qinputs)
    
    qgraph = Embedding(vocabulary_size + 1, hidden_size, mask_zero=False, name="word_embeddings")(qinputs)
    qgraph = Activation(activation="tanh", name="word_activation")(qgraph)
    qgraph = Dropout(rate=dropout_rate, name="word_dropout")(qgraph)
    
    return qgraph, qinputs, qmask


def phrase_embeddings_graph(question_features, dropout_rate, embedding_size=512):
    """
        Three n-gram models are approximate with convolutions of kernel sizes 1, 2, 3. 
        These are applied on the word embeddings with 'same' padding. Then from these 
        convolution results of 'embedding size' the maximum is chosen, which again 
        results in a tensor of 'embedding size' containing the maximum embedding value 
        from the n-grams.
        
        For example, the convolutions may result in three tensors of
            unigram: (1, 6, 512) -> with value -0.06274356 for the first word
            bigram : (1, 6, 512) -> with value -0.31123462 for the first word
            trigram: (1, 6, 512) -> with value  1.3757347 for the first word
        then the maximum value is chosen over all n-grams 
            maximum: (1, 6, 512) -> with value  1.3757347 for the first word
            
        This is similar to 1d pooling, but over an extra dimension (which is not possible with keras). 
        
        A note on the activation:
        It remains unclear, when the original paper is performing the activation.
        In the paper they mention directly at the convolution, but in the implementation
        they perform the activation only after the max pooling. This might be a performance
        tweak, because they choose the maximum values and only have to perform the 
        activation on the maximum values and not each convolution output (reduction by 3).
        Nevertheless, the it remains unclear, if the network would learn a better 
        convolution, when the activation is applied right after instead of a linear one.
    """
    ugraph = Conv1D(embedding_size, kernel_size=1, name="phrase_unigram")(question_features)
    bgraph = Conv1D(embedding_size, kernel_size=2, padding="same", name="phrase_bigram")(question_features)
    tgraph = Conv1D(embedding_size, kernel_size=3, padding="same", name="phrase_trigram")(question_features)
    
    pgraph = Maximum(name="phrase_max")([ugraph, bgraph, tgraph])
    pgraph = Activation("tanh", name="phrase_activation")(pgraph)
    pgraph = Dropout(rate=dropout_rate, name="phrase_dropout")(pgraph)
    
    return pgraph


def phrase_embeddings_model(questions_max_length, dropout_rate, embedding_size=512):
    pinput = Input(shape=(questions_max_length, embedding_size))
    
    pgraph = phrase_embeddings_graph(pinput, dropout_rate, embedding_size)
    
    return Model(inputs=pinput, outputs=pgraph)


def question_embeddings_graph(question_features, dropout_rate, embedding_size=512, question_mask=None):
    """
    A note on the implementation in the paper:
    
    They construct a single LSTM cell (of depth n=num_layers) and then create as many clones
    as the maximum question length. This clone operation duplicates the weights, but during
    training they re-share the weights per layer. Furthermore it is unclear, whether they
    used deep LSTMs or depth of 1 (as noted in the paper), because the default in the 
    implementation is set to 2 [cmd:option('-rnn_layers',2,'number of the rnn layer')]. 
    The re-implementation perform dropout between the deeper levels and between the time steps. 
    Maybe, they re-implemented it also because they have to set the masking manually. 
    
    They also create an output of size
        self.core_output:resize(batch_size, self.seq_length, self.rnn_size):zero()
    but then perform narrowing at each timestep
        self.core_output:narrow(2,t,1):copy(out[self.num_state+1])
    Therefore the question remains, whether they produce only a single output per question?
        The coattention need the features per timestep, therefore there should be an output per word.
        Then only the coattention gives the single attention feature per question. This is maybe
        wrongly pictured in the paper, because there is only a single output in the figure. They say:
        >> The corresponding question-level feature q_s_t is the LSTM hidden vector at time t.
        
    In the original paper they have total number of parameters in ques_level: 5,517,315
        With 2 LSTMs we have 4,198,400 parameters. Why is there a difference? 
        With attention we reach 4,276,292 parameters. When additionally word and phrase embedding
        is considered, then we reach 5,855,812 parameters, which is expected to fit the capacity 
        from the original paper implementation.
    """
    qgraph = LSTM(embedding_size, recurrent_dropout=dropout_rate, return_sequences=True, name="questions_lstm_1")(question_features)
    qgraph = LSTM(embedding_size, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True, name="questions_lstm_2")(qgraph)
    qgraph = Dropout(rate=dropout_rate, name="questions_dropout")(qgraph)
    return qgraph


def question_embeddings_model(questions_max_length, dropout_rate, embedding_size=512, question_mask=None):
    qinput = Input(shape=(questions_max_length, embedding_size))
    
    qgraph = question_embeddings_graph(qinput, dropout_rate, embedding_size, question_mask)
    
    return Model(inputs=qinput, outputs=qgraph)

