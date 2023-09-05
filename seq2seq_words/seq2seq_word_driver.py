import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from seq2seq_word_train import Seq2Seq_Model
from seq2seq_word_preproc import  build_vocabulary,text2vec
from seq2seq_word_inference import decode_sequence 
import argparse 
#from tensorflow.keras.utils import plot_model


max_encoder_seq_len = 8
max_decoder_seq_len = 10
num_encoder_tokens = 100
num_decoder_tokens = 100

def data_vectorize(input_texts, target_texts):

 
    encoder_in_data = np.zeros((len(input_texts), max_encoder_seq_len, num_encoder_tokens), dtype='float32')
    decoder_in_data = np.zeros((len(input_texts), max_decoder_seq_len, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_len, num_decoder_tokens), dtype='float32')

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
    parser.add_argument('-i','--input-file',help='Input file path')
    parser.add_argument('-o','--output-dir',help='Output Dir')
    parser.add_argument('-n','--num-samples',type=int, default=1000,help='Num Samples to use')
    parser.add_argument('-e','--num-epochs',type=int, default=10, help='Num Epochs')

    args = parser.parse_args()

    os.makedirs (args.output_dir,exist_ok=True)
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')


    input_texts = []
    output_texts = []

    for i in range (1, 1000):
        row = lines[i].split('\t')
        input_texts.append (row[0])
        output_texts.append (row[1])

    V =  build_vocabulary(input_texts, output_texts)
     
    V[0].to_csv(args.output_dir + os.sep + "input_vocab.csv")
    V[1].to_csv(args.output_dir + os.sep + "output_vocab.csv")

    vocab_in = V[0]
    vocab_out = V[1]
     

    input_vecs  = text2vec (input_texts, V[0], num_encoder_tokens, 10)
    output_vecs = text2vec (output_texts, V[1], num_decoder_tokens, 10)

    #print(input_vecs)
    #sys.exit()


    encoder_in_data, decoder_in_data, decoder_target_data =  data_vectorize(input_vecs, output_vecs)

    M = Seq2Seq_Model(args.output_dir, num_encoder_tokens, num_decoder_tokens)

    # Model Summary
    print(M.model.summary())

    print("encoder_in_data shape:", encoder_in_data.shape)
    print("decoder_in_data shape:", decoder_in_data.shape)
    print("decoder_target_data shape:", decoder_target_data.shape)

    #sys.exit()
    # Visuaize the model
    #plot_model(M.model, show_shapes=True)
    plt.show()

    hist = M.fit_model(encoder_in_data, decoder_in_data, decoder_target_data, args.num_epochs)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(hist.history['loss'], label='Training Loss')
    axs[0].plot(hist.history['val_loss'], label='Validation Loss')
    axs[0].legend()
    axs[1].plot(hist.history['accuracy'],label='Training Accurcay')
    axs[1].plot(hist.history['val_accuracy'],label='Validation Accurcay')
    axs[1].legend()

    plt.show()

    for seq_index in range(10):
        input_seq = encoder_in_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(P, M, input_token_id, target_token_id, input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)

