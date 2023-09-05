import numpy as np
from collections import Counter 
import re
import pandas as pd

def build_vocabulary(input_data, output_data):
     V = []
     for data in [input_data, output_data]:
        tokens = " ".join (data).split(" ")
        vocab = Counter (tokens)
        token_dict = dict (vocab)
        df = pd.DataFrame (columns=['word','count'])
        df ['word'] = list (token_dict.keys())
        df ['count'] = list (token_dict.values())
        df = df.sort_values (by=['count'], ascending=False, ignore_index=True)
        V.append (df)

     return V                          
    


def text2vec (input_text, df_vocab, vocab_size, vec_len):
    D = df_vocab.iloc[:vocab_size]
    ids = [i for i in range (0, len (D))]
    D = D.assign (id=ids)
    D.index = D['word'].to_list()


    text_vecs = []
    for text in input_text:
        words = text.split(" ")
        words = [w for w in words if w in D.index]
        vec = [D.loc[w]['id'] for w in words]
        text_vecs.append (vec)
    return text_vecs     


