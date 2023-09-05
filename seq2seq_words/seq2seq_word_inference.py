import numpy as np

def get_lookup (df):
    token2id = {}
    id2token = {}
    count = 0
    prob = []
    counts = df['count'].to_list()
    norm = np.sum (counts)
    prob = [p /norm for p in counts]
    
    for index, row in df.iterrows():
        token2id [row['word']] = count
        id2token [count] = row['word']
        count+=1
    return token2id, id2token, prob      


def decode_sequence(config, M, vocab_in, vocab_out, input_seq):

    num_encoder_tokens = config.getint('Params','num_encoder_tokens')
    num_decoder_tokens = config.getint('Params','num_decoder_tokens')
    
    max_encoder_seq_len = config.getint('Params','max_encoder_vec_len')
    max_decoder_seq_len = config.getint('Params','max_decoder_vec_len')


    enc_token2id, enc_id2token, enc_prob = get_lookup (vocab_in)
    dec_token2id, dec_id2token, dec_prob = get_lookup (vocab_out)
 
    #Encode the input as state vectors.
    states_value = M.encoder_model.predict(input_seq)

    #Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    #Get the first character of target sequence with the start character.
    target_seq[0, 0, dec_token2id['__START__']] = 1.

    #Sampling loop for a batch of sequences
    #(to simplify, here we assume a batch of size 1).
    
    stop_condition = False
    decoded_sentence = ' '
    
    while not stop_condition:
        output_tokens, h, c = M.decoder_model.predict([target_seq] + states_value)

        preds  =  output_tokens[0, 0,:]
        preds  = [preds[i] / dec_prob[i] for i in range (0, len (preds))]
        #print("preds=",preds)
        
        #Sample a token
        sampled_token_index = np.argmax(preds)
        
        sampled_word = dec_id2token[sampled_token_index]
        if sampled_word not in decoded_sentence.split(" ") or np.random.random([10])[0] > 0.25:  
            decoded_sentence = decoded_sentence + "  " +  sampled_word
        
            
            
        print("decoced sent:",decoded_sentence)

        if len(decoded_sentence.split(" ")) > max_decoder_seq_len :
            stop_condition = True 
        if sampled_word == '__STOP__':
            stop_condition = True 

        #Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        #Update states
        states_value = [h, c]

    return decoded_sentence
