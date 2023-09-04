import sys
import os
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import *
from tensorflow.keras.initializers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class Seq2Seq_Model:
    def __init__(self, workspace, num_encoder_tokens, num_decoder_tokens):

        latent_dim = 50
        self.workspace = workspace
        self.model_dir = self.workspace + os.sep + "trained_model"
        self.log_dir = self.workspace + os.sep + "log"

        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Define and process the input sequence
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Using `encoder_states` set up the decoder as initial state.
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Final model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Model Summary
        print(self.model.summary())

        # Define sampling models
        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def fit_model(self, encoder_in_data, decoder_in_data, decoder_target_data, nepochs):
        batch_size = 64

        chkpt = ModelCheckpoint(filepath=self.model_dir + os.sep + "model.hdf5",
                                save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

        tboard = TensorBoard(log_dir=self.log_dir)

        callbacks = [chkpt, tboard]

        # Compiling and training the model
        self.model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001),
                           loss='categorical_crossentropy')

        hist = self.model.fit([encoder_in_data, decoder_in_data],
                              decoder_target_data, callbacks=callbacks, batch_size=batch_size, epochs=nepochs,
                              validation_split=0.2)

        return hist

