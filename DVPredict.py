from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import model_from_json


import pickle


import numpy as np


def predict(tweet):
    
    # load tokenizer
    with open('tokenizer.pickle', 'rb') as token:
        tokenizer = pickle.load(token)
    with open('max_seq_len.txt', 'r') as maxseq:
        MAX_SEQUENCE_LENGTH = maxseq.read()
        print()
        MAX_SEQUENCE_LENGTH = int(MAX_SEQUENCE_LENGTH)
        
        
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # load weights into new model
    model.load_weights("model.h5")
    
    test_sequence = tokenizer.texts_to_sequences([tweet])
    test_sen = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = model.predict(test_sen)
    labels = [1, 0]
    pred = labels[np.argmax(prediction)]
    
    # prediction
    if pred == 1:
        print('Yes')
    else:
        print('No')
        
    return result

if __name__ == '__main__':
    tweet = input('Enter tweet : ')
    predict(tweet)