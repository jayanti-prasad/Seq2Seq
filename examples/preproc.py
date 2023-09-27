from collections import Counter
import pandas  as pd

def build_vocabulary(text_data):
    """
    We can build the vocabulary in terms of a data frame which have the following columns:
    - id
    - word
    - count

    """
    corpus = " ".join(text_data)
    tokens = corpus.split(" ")
    word_counter = Counter(tokens)
    Vocab = dict(word_counter)
    df = pd.DataFrame(columns=['word', 'count'])
    df['word'] = list(Vocab.keys())
    df['count'] = list(Vocab.values())
    df = df.sort_values(by=['count'], ignore_index=True, ascending=False)
    print("vocab:", df.shape, df.columns)
    return df


class Vectorizer:
    """
    Vectorizer used the vocabulary data frame and applies a cutoff on the vocabulary size.

    """

    def __init__(self, df_vocab, cut_off):
        df = df_vocab[df_vocab['count'] > cut_off]
        keys = df['word'].to_list()
        values = [i for i in range(0, len(keys))]
        self.word2id = dict(zip(keys, values))

    def text2vec(self, text):
        tokens = text.split(" ")
        vec = [self.word2id[token] for token in tokens if token in self.word2id]
        return vec


