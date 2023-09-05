import numpy as np


def decode_sequence(P, M, input_token_id, target_token_id, input_seq):
    
    num_encoder_tokens = P['num_encoder_tokens']
    num_decoder_tokens = P['num_decoder_tokens']
    
    max_encoder_seq_len = P['max_encoder_seq_len']
    max_decoder_seq_len = P['max_decoder_seq_len']
    
    reverse_input_char_index = dict((i, char) for char, i in input_token_id.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_id.items())

    
    #Encode the input as state vectors.
    states_value = M.encoder_model.predict(input_seq)

    #Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    #Get the first character of target sequence with the start character.
    target_seq[0, 0, target_token_id['\t']] = 1.


    #Sampling loop for a batch of sequences
    #(to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = M.decoder_model.predict([target_seq] + states_value)

        #Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        #Exit condition: either hit max length
        #or find stop character.
        if (sampled_char == '\n' or
            len(decoded_sentence) > max_decoder_seq_len):
            stop_condition = True
    
        #Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        #Update states
        states_value = [h, c]

    return decoded_sentence
