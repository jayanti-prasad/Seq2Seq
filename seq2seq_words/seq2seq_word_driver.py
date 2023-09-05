import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from seq2seq_word_train import Seq2Seq_Model
from seq2seq_word_preproc import  build_vocabulary,text2vec
from seq2seq_word_inference import decode_sequence 
import argparse 
import configparser 
#from tensorflow.keras.utils import plot_model


def data_vectorize(input_texts, target_texts):

    num_encoder_tokens = config.getint('Params','num_encoder_tokens')
    num_decoder_tokens = config.getint('Params','num_decoder_tokens')
    max_encoder_vec_len = config.getint('Params','max_encoder_vec_len')
    max_decoder_vec_len = config.getint('Params','max_decoder_vec_len')
    
    encoder_in_data = np.zeros((len(input_texts), max_encoder_vec_len, num_encoder_tokens), dtype='float32')
    decoder_in_data = np.zeros((len(input_texts), max_decoder_vec_len, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_vec_len, num_decoder_tokens), dtype='float32')

    for i in range (0, len (input_texts)):
       if  (len (input_texts[i]) > 0)  & (len (target_texts[i]) > 0): 
           for j, token_id in enumerate (input_texts[i]):
               encoder_in_data[i, j, token_id] = 1
           for j, tokein_id in enumerate (target_texts[i][1:]):
               decoder_in_data[i, j, token_id] = 1
           for j, token_id in enumerate (target_texts[i][:-1]):
               decoder_target_data[i, j, token_id] = 1

    return encoder_in_data, decoder_in_data, decoder_target_data




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()

    parser.add_argument('-i','--config-file',help='Input file path')
    args = parser.parse_args()

    config.read(args.config_file)

    num_encoder_tokens = config.getint('Params','num_encoder_tokens')
    num_decoder_tokens = config.getint('Params','num_decoder_tokens')
    max_encoder_vec_len = config.getint('Params','max_encoder_vec_len')
    max_decoder_vec_len = config.getint('Params','max_decoder_vec_len')
    num_epochs = config.getint('Params','num_epochs')
    output_dir = config['Params']['output_dir']

    os.makedirs (config['Params']['output_dir'],exist_ok=True)
    
    with open(config['Params']['input_file'], 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')


    input_texts = []
    output_texts = []

    for i in range (1, config.getint ('Params','num_data_points')):
        row = lines[i].split('\t')
        if row[0] not in input_texts:
           input_texts.append (row[0])
           output_texts.append ("__START__ " + row[1] + " __STOP__")

    print("number of data points:", len(input_texts))

    [vocab_in, vocab_out ] =  build_vocabulary(input_texts, output_texts)
     
    vocab_in.to_csv(output_dir + os.sep + "input_vocab.csv")
    vocab_out.to_csv(output_dir + os.sep + "output_vocab.csv")

    input_vecs  = text2vec (input_texts, vocab_in,  num_encoder_tokens, max_encoder_vec_len)
    output_vecs = text2vec (output_texts, vocab_out, num_decoder_tokens, max_decoder_vec_len)

    encoder_in_data, decoder_in_data, decoder_target_data =  data_vectorize(input_vecs, output_vecs)

    M = Seq2Seq_Model(config)

    # Model Summary
    print(M.model.summary())

    print("encoder_in_data shape:", encoder_in_data.shape)
    print("decoder_in_data shape:", decoder_in_data.shape)
    print("decoder_target_data shape:", decoder_target_data.shape)

    #sys.exit()
    # Visuaize the model
    #plot_model(M.model, show_shapes=True)
    plt.show()

    print("num_epcohs")
    hist = M.fit_model(encoder_in_data, decoder_in_data, decoder_target_data,num_epochs)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(hist.history['loss'], label='Training Loss')
    axs[0].plot(hist.history['val_loss'], label='Validation Loss')
    axs[0].legend()
    axs[1].plot(hist.history['accuracy'],label='Training Accurcay')
    axs[1].plot(hist.history['val_accuracy'],label='Validation Accurcay')
    axs[1].legend()

    plt.show()

    for seq_index in [10, 50, 100, 103, 200, 304]:
        input_seq = encoder_in_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(config, M, vocab_in, vocab_out, input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)

