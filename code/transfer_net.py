import tensorflow.keras as keras
import tensorflow as tf

def load_freeze_transfer(output_directory_old, new_nb_classes, lr = 0.00001):
  
  model_path = output_directory_old + 'best_model.hdf5'
  model = keras.models.load_model(model_path)
  
  model_new = keras.Sequential()
		
  for layer in model.layers[:-1]: # just exclude last layer from copying
	  model_new.add(layer)
  for layer in model_new.layers:
    layer.trainable = False
  if new_nb_classes > 2:
    model_new.add(keras.layers.Dense(new_nb_classes, activation='softmax'))
    model_new.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(lr = lr), 
  		metrics=['accuracy'])
  else:
  	model_new.add(keras.layers.Dense(1, activation='sigmoid'))
    model_new.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(lr = lr), 
  		metrics=['accuracy'])

  return model_new
