import matplotlib.pyplot as plt
from seq2seq_char_train import Seq2Seq_Model
from seq2seq_char_preproc import get_data  
from seq2seq_char_preproc import data_vectorize 
from seq2seq_char_inference import decode_sequence 
import argparse 
#from tensorflow.keras.utils import plot_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-file',help='Input file path')
    parser.add_argument('-o','--output-dir',help='Output Dir')
    parser.add_argument('-n','--num-samples',type=int, default=1000,help='Num Samples to use')
    parser.add_argument('-e','--num-epochs',type=int, default=10, help='Num Epochs')

    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    P, input_token_id, target_token_id, input_texts, target_texts = get_data(lines, args.num_samples)

    print(P['input_chars'])
    print(P['target_chars'])
    # sys.exit()

    encoder_in_data, decoder_in_data, decoder_target_data = data_vectorize(P, input_texts, target_texts, input_token_id,
                                                                           target_token_id)

    M = Seq2Seq_Model(args.output_dir, P['num_encoder_tokens'], P['num_decoder_tokens'])

    # Model Summary
    print(M.model.summary())

    print("encoder_in_data shape:", encoder_in_data.shape)
    print("decoder_in_data shape:", decoder_in_data.shape)
    print("decoder_target_data shape:", decoder_target_data.shape)

    # Visuaize the model
    #plot_model(M.model, show_shapes=True)
    plt.show()

    hist = M.fit_model(encoder_in_data, decoder_in_data, decoder_target_data, 15)

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

