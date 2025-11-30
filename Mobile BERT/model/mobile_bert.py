import tensorflow_hub as hub
import tensorflow

from tensorflow.python.keras.layers import Input, Dropout, Dense
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import EarlyStopping

from facades.log import Log

class MobileBERT:

    def __init__(self):
        self.model_name = "mobile_bert.202511261238"
        
        self.input = Input(shape=(), dtype=tensorflow.string, name = "input_text")
        
        # Pre-Processing & Vectorizer Layer
        self.preprocessing_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", "preprocessor")
        
        self.encoder_layer = hub.KerasLayer("https://tfhub.dev/google/mobilebert/1", trainable = True, name = "encoder")

        # Fully Connected
        self.dropout = Dropout(0.1)
        self.output = Dense(1, activation='sigmoid', name='output')
        Log.info(f"Initialized {self.model_name} model")

    # Early stopping after loss are not improved for some epochs
    def _callback_early_stopping(self):
        early_stopping_tolerance = 5
        return EarlyStopping(
            monitor = "val_loss",
            patience = early_stopping_tolerance,
            restore_best_weights = True
        )

    
    def init_build(self):
        encoder_input = self.preprocessing_layer(input)

        encoder_output = self.encoder_layer(encoder_input)
        
        network = self.dropout(encoder_output["pooled_output"])
        network = self.output(network)

        model = Model(inputs=self.input, outputs=network, name=self.model_name)

        return model

    def train(self):
        model = self.init_build()
        model.compile(
            optimizer = tensorflow.keras.optimizers.Adam(learning_rate=3e-5),
            loss = tensorflow.keras.losses.BinaryCrossentropy(),
            metrics = [tensorflow.keras.metrics.BinaryAccuracy(name='accuracy')]
        )

        Log.info(f"Compiled {self.model_name} model")

        return model