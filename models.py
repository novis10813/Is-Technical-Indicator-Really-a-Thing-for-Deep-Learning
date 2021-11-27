import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

class MLP_model:
    def __init__(self, window_size, features):
        self.window_size = window_size
        self.features = features
    
    @property
    def model(self):
        input = tf.keras.Input(shape=(self.window_size, self.features), name='input_layer')
        x = layers.Flatten()(input)
        x = layers.Dense(300, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = layers.Dropout(0.7)(x)
        output = layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        model = Model(input, output, name='MLP_model')
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
class CDT_1D_model:
    def __init__(self, window_size, features):
        self.window_size = window_size
        self.features = features
        
    def __CNN_blocks(self, id, filters, kernel_size, pool_size, pool_strides):
        block = tf.keras.Sequential([
            tf.keras.layers.Convolution2D(filters=filters, kernel_size=(1, kernel_size), strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal'),
            tf.keras.layers.MaxPool2D(pool_size=(1, pool_size), strides=(1, pool_strides), padding='valid')
        ], name=f'feature_extractor_{id}')
    
        return block

    @property
    def model(self):
        input = tf.keras.Input(shape=(self.window_size, self.features), name='input_layer')
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3), name='add_dim')(input)
        x = tf.keras.layers.Permute((2, 1, 3), name='feature_time_transpose')(x)
        x = self.__CNN_blocks(id=1, filters=32, kernel_size=4, pool_size=4, pool_strides=4)(x)
        x = self.__CNN_blocks(id=2, filters=64, kernel_size=3, pool_size=3, pool_strides=3)(x)
        x = self.__CNN_blocks(id=3, filters=128, kernel_size=2, pool_size=2, pool_strides=2)(x)
        # x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2), name='reduce_dim')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = tf.keras.layers.Dropout(0.7)(x)
        x = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer='random_normal')(x)
        x = tf.keras.layers.Dropout(0.7)(x)
        output = tf.keras.layers.Dense(3, activation='softmax')(x)
        model = tf.keras.Model(input, output, name='CDT-1D_model')

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model