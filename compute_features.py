import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from math import log
from itertools import product
import os
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import jaccard_distance
import wikipediaapi
import warnings
import re

from nltk.corpus import wordnet_ic
from multiprocessing import Pool, cpu_count, freeze_support

# --- Definición de funciones globales ---

def load_dataset(directory, dataset_category='train'):
    file_inputs, file_gs = [], []
    folder_path = os.path.join(directory, 'train' if dataset_category == 'train' else 'test-gold')

    file_inputs = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.startswith('STS.input') or fname.startswith('STS.input.surprise')
    ]

    file_gs = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if (fname.startswith('STS.gs') or fname.startswith('STS.gs.surprise')) and not fname.endswith('ALL.txt')
    ]

    combined_data = []
    for input_path, gs_path in zip(sorted(file_inputs), sorted(file_gs)):
        base_name = os.path.basename(input_path).split('.')[2]
        if base_name.startswith('surprise'):
            base_name = '.'.join(os.path.basename(input_path).split('.')[2:4])
        with open(input_path, 'r', encoding='utf-8') as inp_file, open(gs_path, 'r', encoding='utf-8') as gs_file:
            input_pairs = [line.strip().split('\t') for line in inp_file]
            gs_scores = [float(line.strip()) for line in gs_file]
            combined_data.extend(
                [
                    (pair[0], pair[1], score, base_name)
                    for pair, score in zip(input_pairs, gs_scores)
                ]
            )
    return combined_data

def remove_punctuation_tokens(tokens):
    return [token for token in tokens if token not in string.punctuation]

wnl = nltk.stem.WordNetLemmatizer()

def lemmatize_WNL(p, warn=False):
    d = {'NN': 'n', 'NNS': 'n', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v', 'RB': 'r', 'RBR': 'r', 'RBS': 'r'}
    if p[1] in d:
        return wnl.lemmatize(p[0], pos=d[p[1]])
    else:
        return p[0]

def lemmatize(row, name, warn=False):
    tokens = row[name]
    tokens = [token.lower() for token in tokens]
    pair_tags = nltk.pos_tag(tokens)
    return [lemmatize_WNL(pair, warn) for pair in pair_tags]

def compute_pathlen_similarity(list1, list2):
    pathlen_scores = []
    for word1 in list1:
        for word2 in list2:
            synsets1 = wn.synsets(word1)
            synsets2 = wn.synsets(word2)
            if synsets1 and synsets2:
                scores = [
                    s1.path_similarity(s2)
                    for s1, s2 in product(synsets1, synsets2)
                    if s1.path_similarity(s2) is not None
                ]
                if scores:
                    pathlen_scores.append(max(scores))
    if pathlen_scores:
        average_similarity = sum(pathlen_scores) / len(pathlen_scores)
    else:
        average_similarity = 0
    return average_similarity

def compute_lin_similarity(list1, list2, ic):
    lin_scores = []
    for word1 in list1:
        for word2 in list2:
            synsets1 = wn.synsets(word1)
            synsets2 = wn.synsets(word2)
            if synsets1 and synsets2:
                scores = []
                for s1, s2 in product(synsets1, synsets2):
                    try:
                        similarity = s1.lin_similarity(s2, ic)
                        if similarity is not None:
                            scores.append(similarity)
                    except:
                        continue
                if scores:
                    lin_scores.append(max(scores))
    if lin_scores:
        average_similarity = sum(lin_scores) / len(lin_scores)
    else:
        average_similarity = 0
    return average_similarity

"""
def process_chunk(chunk_and_ic):
    chunk, ic = chunk_and_ic
    chunk['pathlen_similarity'] = chunk.apply(
        lambda row: compute_pathlen_similarity(row["sentence_lemmas_0"].split(), row["sentence_lemmas_1"].split()), axis=1)
    chunk['lin_similarity'] = chunk.apply(
        lambda row: compute_lin_similarity(row["sentence_lemmas_0"].split(), row["sentence_lemmas_1"].split(), ic), axis=1)
    return chunk[['pathlen_similarity', 'lin_similarity']]
"""

def process_chunk(chunk_and_ic):
    chunk, ic = chunk_and_ic
    chunk['pathlen_similarity'] = chunk.apply(
        lambda row: compute_pathlen_similarity(row["lemmas_0"], row["lemmas_1"]), axis=1)
    return chunk[['pathlen_similarity']]


def parallel_process(df, ic, num_chunks):
    chunks = np.array_split(df, num_chunks)
    chunks_with_ic = [(c, ic) for c in chunks]

    with Pool(num_chunks) as pool:
        results = pool.map(process_chunk, chunks_with_ic)
    return pd.concat(results, axis=0)


if __name__ == "__main__":
    # Esta línea es clave en Windows para scripts con multiprocessing
    freeze_support()

    # Descargas NLTK dentro del main
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet_ic', quiet=True)

    stopwords_set = set(stopwords.words('english'))

    # Cargar el IC dentro del main
    ic = wordnet_ic.ic('ic-brown.dat')

    data_dir = './datasets/'  # Ajusta el path según tu entorno
    train_data = load_dataset(data_dir, dataset_category='train')
    columns = ['sentence_0', 'sentence_1', 'score', 'dataset_name']
    train_data = pd.DataFrame(train_data, columns=columns)

    test_data = load_dataset(data_dir, dataset_category='test')
    test_data = pd.DataFrame(test_data, columns=columns)

    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of test samples: {len(test_data)}")

    train_data['tokens_0'] = train_data['sentence_0'].apply(nltk.word_tokenize)
    train_data['tokens_1'] = train_data['sentence_1'].apply(nltk.word_tokenize)
    train_data['tokens_0'] = train_data['tokens_0'].apply(remove_punctuation_tokens)
    train_data['tokens_1'] = train_data['tokens_1'].apply(remove_punctuation_tokens)
    train_data['lemmas_0'] = train_data.apply(lambda row: lemmatize(row, "tokens_0", True), axis=1)
    train_data['lemmas_1'] = train_data.apply(lambda row: lemmatize(row, "tokens_1", True), axis=1)
    train_data['sentence_lemmas_0'] = train_data.apply(lambda row: " ".join(row["lemmas_0"]), axis=1)
    train_data['sentence_lemmas_1'] = train_data.apply(lambda row: " ".join(row["lemmas_1"]), axis=1)

    test_data['tokens_0'] = test_data['sentence_0'].apply(nltk.word_tokenize)
    test_data['tokens_1'] = test_data['sentence_1'].apply(nltk.word_tokenize)
    test_data['tokens_0'] = test_data['tokens_0'].apply(remove_punctuation_tokens)
    test_data['tokens_1'] = test_data['tokens_1'].apply(remove_punctuation_tokens)
    test_data['lemmas_0'] = test_data.apply(lambda row: lemmatize(row, "tokens_0", True), axis=1)
    test_data['lemmas_1'] = test_data.apply(lambda row: lemmatize(row, "tokens_1", True), axis=1)
    test_data['sentence_lemmas_0'] = test_data.apply(lambda row: " ".join(row["lemmas_0"]), axis=1)
    test_data['sentence_lemmas_1'] = test_data.apply(lambda row: " ".join(row["lemmas_1"]), axis=1)

    # Si colapsa hasta con 1 chunk, prueba a no dividir: num_chunks = 1
    # Y también Pool(1) en parallel_process
    num_chunks = cpu_count() - 1

    print("Processing training data...")
    features_train = parallel_process(train_data, ic, num_chunks)
    features_train.to_csv("features_train.csv", index=False)

    print("Processing test data...")
    features_test = parallel_process(test_data, ic, num_chunks)
    features_test.to_csv("features_test.csv", index=False)

    print("Done!")