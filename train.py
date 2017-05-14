from text_cnn import TextCNN
import numpy as np
import keras

from keras.datasets import reuters
from keras.preprocessing import sequence
from keras.models import Model

import json

with open('./config.json', 'r') as f:
  config = json.load(f)

(x_train, y_train),(x_test, y_test) = reuters.load_data(num_words=config['vocabulary_size'], test_split=0.2)

config['num_class'] = np.max(y_train)+1

maxlen_train = len(max(x_train, key=len))+1
maxlen_test = len(max(x_test, key=len))+1

config['max_length'] = maxlen_train if maxlen_train>maxlen_test else maxlen_test 

x_train = sequence.pad_sequences(x_train, maxlen=config['max_length'])
x_test = sequence.pad_sequences(x_test, maxlen=config['max_length'])

y_train = keras.utils.to_categorical(y_train, config['num_class'])
y_test = keras.utils.to_categorical(y_test, config['num_class'])

print(json.dumps(config))

cnn = TextCNN(config)

history = cnn.model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=1,
    verbose=1,
    validation_data=(x_test,y_test)
)

score = cnn.model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print('eval score: ', score[0])
print('eval accuracy: ', score[1])

cnn.model.save_weights('/home/andy/Desktop/models/insurance_faq.h5')
modelw = open('/home/andy/Desktop/models/insurance_faq.config.json', 'w')
modelw.write(cnn.model.to_json())

cnn.model.save('./reuter_model')

#use average pooling for word vector convolution
