import numpy as np
import re

def get_data(lines, num_samples):

    input_chars = set()
    target_chars = set()
    input_texts = []
    target_texts = []

    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text, extra = line.split('\t')
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)

        for char in input_text:
            if char not in input_chars:
                input_chars.add(char)
        for char in target_text:
            if char not in target_chars:
                target_chars.add(char)

    input_chars = sorted(list(input_chars))
    target_chars = sorted(list(target_chars))

    num_encoder_tokens = len(input_chars)
    num_decoder_tokens = len(target_chars)
    max_encoder_seq_len = max([len(txt) for txt in input_texts])
    max_decoder_seq_len = max([len(txt) for txt in target_texts])

    P = {'num_encoder_tokens': num_encoder_tokens,
         'num_decoder_tokens': num_decoder_tokens,
         'max_encoder_seq_len': max_encoder_seq_len,
         'max_decoder_seq_len': max_decoder_seq_len,
         'input_chars': input_chars,
         'target_chars': target_chars}

    # Define data for encoder and decoder
    input_token_id = dict([(char, i) for i, char in enumerate(input_chars)])
    target_token_id = dict([(char, i) for i, char in enumerate(target_chars)])

    return P, input_token_id, target_token_id, input_texts, target_texts


def data_vectorize(P, input_texts, target_texts, input_token_id, target_token_id):
    num_encoder_tokens = P['num_encoder_tokens']
    num_decoder_tokens = P['num_decoder_tokens']

    max_encoder_seq_len = P['max_encoder_seq_len']
    max_decoder_seq_len = P['max_decoder_seq_len']
    input_chars = P['input_chars']
    target_chars = P['target_chars']

    encoder_in_data = np.zeros((len(input_texts), max_encoder_seq_len, num_encoder_tokens), dtype='float32')
    decoder_in_data = np.zeros((len(input_texts), max_decoder_seq_len, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_len, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_in_data[i, t, input_token_id[char]] = 1.

        for t, char in enumerate(target_text):
            decoder_in_data[i, t, target_token_id[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_id[char]] = 1.

    return encoder_in_data, decoder_in_data, decoder_target_data
