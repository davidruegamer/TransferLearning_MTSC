import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 

class Classifier_FCN:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, 
	    lr = 0.00001, filters=128, patience=50, monitor_metric='val_loss', callbacks=[]):
     
		self.output_directory = output_directory
        
		if build == True:
			self.model = self.build_model(input_shape, nb_classes, lr, filters, callbacks, monitor_metric, patience)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory+'model_init.hdf5')
		return

	def build_model(self, input_shape, nb_classes, lr, filters, callbacks, monitor_metric, patience):
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=filters, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=filters*2, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(filters, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    if (nb_classes > 2):
  		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
  		model = keras.models.Model(inputs=input_layer, outputs=output_layer)
  		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(lr = lr), 
  			metrics=['accuracy'])
		else:
			output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)
  		model = keras.models.Model(inputs=input_layer, outputs=output_layer)
  		model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(lr = lr), 
  			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		es = tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=patience, verbose=0,
											  restore_best_weights=True)

		self.callbacks = [es,reduce_lr,model_checkpoint] + callbacks

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true, batch_size = 16, nb_epochs = 2000):
		
		# mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 
		
		if x_val is None or y_val is None:
			val_data = None
			self.callbacks = self.callbacks[1:] # no early stopping
		else:
        		val_data = (x_val, y_val)


		hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=val_data, callbacks=self.callbacks)
		
		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.hdf5')

		# model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		# y_pred = model.predict(x_val)
		
		# save predictions
        # np.save(self.output_directory + 'y_pred.npy', y_pred)

		# convert the predicted from binary to integer 
		# y_pred = np.argmax(y_pred , axis=1)

		keras.backend.clear_session()
		
		return hist.history

	def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
		model_path = self.output_directory + 'best_model.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
		return y_pred
		
		
