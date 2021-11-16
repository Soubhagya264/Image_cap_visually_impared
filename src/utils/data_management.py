import logging
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
def data_generator(train_descriptions, train_features, wordtoix, max_length,vocab_size):
            X1, X2, y = list(), list(), list()
            for key, des_list in train_descriptions.items():
                pic = train_features[key + '.jpg']
                for cap in des_list:
                    seq = [wordtoix[word] for word in cap.split(' ') if word in wordtoix]
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
                        out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
                        # store
                        X1.append(pic)
                        X2.append(in_seq)
                        y.append(out_seq)
            X2 = np.array(X2)
            X1 = np.array(X1)
            y = np.array(y)  
            return X1,X2,y          
                    