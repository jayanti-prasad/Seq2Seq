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



if __name__ == "__main__":

     input_voc_size = 10000
     output_voc_size = 10000
     
     input_file = r"C:\Users\jayanti.prasad\Data\seq2seq_data\fra.txt"
     
     with open(input_file, 'r', encoding='utf-8') as f:
         lines = f.read().split('\n')

     input_texts = []
     output_texts = []

     for i in range (1, 1000):
         row = lines[i].split('\t')
         input_texts.append (row[0])
         output_texts.append ("__START__ " + row[1] + " __STOP__")

     V =  build_vocabulary(input_texts, output_texts)
     V[0].to_csv("input_vocab.csv")
     V[1].to_csv("output_vocab.csv")
     

     input_vecs  = text2vec (input_texts, V[0], 10000, 10)
     output_vecs = text2vec (output_texts, V[1], 10000, 10)

     for i in range (0, len(input_vecs)):
          print(i, input_vecs[i], output_vecs[i])
          
                  
