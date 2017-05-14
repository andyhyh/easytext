import keras
from keras.layers import Input,Dense,Dropout,Conv1D,Embedding,MaxPooling1D,Flatten
from keras.models import Model

class TextCNN(object):

  def validate_config(self, config):
    valid = True
    for layer in config['cnn_layers']:
      if not (layer['depth'] == len(layer['filters']) 
              and layer['depth'] == len(layer['kernels']) 
              and layer['depth'] == len(layer['poolings'])
              and layer['depth'] == len(layer['dropouts'])):
        print("conv layers failed to validate")
        return False
    if not (config['dense_layers']['depth'] == len(config['dense_layers']['size'])
            and config['dense_layers']['depth'] == len(config['dense_layers']['dropouts'])):
      print("fully connected layers failed to validate")
      return False
    return True

  def __init__(self, config):
    
    if not self.validate_config(config):
      print('validation failed')
      return
    input = Input(shape=(config['max_length'],), name='input')
    
    #input may be used for different channels - for now just 1 embedding channel
    
    embed = Embedding(
      input_dim=config['vocabulary_size'],
      output_dim=config['embedding_size'],
      input_length=config['max_length']
    )(input)

    conv_layers = []

    print('num: ',len(config['cnn_layers']))

    layer_id = 0

    #build conv layers
    for layers in config['cnn_layers']:

      entry_layer = embed
      
      for index in xrange(layers['depth']):
        c = Conv1D(
          filters = layers['filters'][index],
          kernel_size = layers['kernels'][index],
          strides = 1,
          use_bias = 1,
          activation = 'relu',
          name='conv_'
                +'_k'+str(layers['kernels'][index])
                +'_f'+str(layers['filters'][index])
                +'_idx'+str(index)
                +'_id'+str(layer_id)
        )(entry_layer)

        p = MaxPooling1D(
          pool_size = layers['poolings'][index], 
          name='pool_p'+str(layers['poolings'][index])+'_id'+str(layer_id)
        )(c)

        d = Dropout(
          rate = layers['dropouts'][index], 
          name='dropout_r'+str(layers['dropouts'][index])+'_id'+str(layer_id)
        )(p)
        entry_layer = d
        layer_id = layer_id+1

      #flatten last layer
      f = Flatten()(entry_layer)
      conv_layers.append(f)
    
    concat = keras.layers.concatenate(conv_layers)

    #dense layers
    next_entry = concat
    for index in xrange(config['dense_layers']['depth']):
      drop = Dropout(rate=config['dense_layers']['dropouts'][index])(next_entry)
      dense = Dense(units=config['dense_layers']['size'][index], activation='relu')(drop)
      next_entry = dense

    out = Dense(config['num_class'], activation='softmax')(next_entry)

    self.model = Model(inputs=[input], outputs=[out])

    self.model.compile(
      loss=keras.losses.categorical_crossentropy,
      optimizer=keras.optimizers.Adam(),
      metrics=['accuracy']
    )