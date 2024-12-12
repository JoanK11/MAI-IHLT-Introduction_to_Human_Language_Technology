import pandas as pd
from statistics import harmonic_mean

from scripts.lexical_features import *
from scripts.syntactic_features import *


def compute_features(data, stopwords, nltk, wn, ic, word_freq, total_freq):
    features = pd.DataFrame()

    # ============================
    # Lexical Features
    # ============================

    # Similarity Features
    features['longest_common_substring'] = data.apply(lambda row: longest_common_substring(row["sentence_lemmas_0"], row["sentence_lemmas_1"]), axis=1)
    features['longest_common_subsequence'] = data.apply(lambda row: longest_common_subsequence(row["sentence_lemmas_0"], row["sentence_lemmas_1"]), axis=1)
    features['greedy_string_tiling'] = data.apply(lambda row: greedy_string_tiling(row["sentence_lemmas_0"], row["sentence_lemmas_1"], min_match_length=1), axis=1)

    # Character n-gram Similarity Features
    features['2_gram_char'] = data.apply(lambda row: similarity_char_ngrams(row["lemmas_0"], row["lemmas_1"], 2), axis=1)
    features['3_gram_char'] = data.apply(lambda row: similarity_char_ngrams(row["lemmas_0"], row["lemmas_1"], 3), axis=1)
    features['4_gram_char'] = data.apply(lambda row: similarity_char_ngrams(row["lemmas_0"], row["lemmas_1"], 4), axis=1)

    # Word n-gram Jaccard Similarity Features
    features['1_gram_word_Jaccard'] = data.apply(lambda row: similarity_words_ngrams_jaccard(row["lemmas_0"], row["lemmas_1"], 1, nltk), axis=1)
    features['3_gram_word_Jaccard'] = data.apply(lambda row: similarity_words_ngrams_jaccard(row["lemmas_0"], row["lemmas_1"], 3, nltk), axis=1)
    features['4_gram_word_Jaccard'] = data.apply(lambda row: similarity_words_ngrams_jaccard(row["lemmas_0"], row["lemmas_1"], 4, nltk), axis=1)

    # Word n-gram Jaccard Similarity Features without Stopwords
    features['2_gram_word_Jaccard_without_SW'] = data.apply(lambda row: similarity_words_ngrams_jaccard(row["lemmas_0"], row["lemmas_1"], 2, nltk, stopwords), axis=1)
    features['4_gram_word_Jaccard_without_SW'] = data.apply(lambda row: similarity_words_ngrams_jaccard(row["lemmas_0"], row["lemmas_1"], 4, nltk, stopwords), axis=1)

    # Word n-gram Containment Similarity Features without Stopwords (a)
    features['1_gram_word_Containment_without_SW_a'] = data.apply(lambda row: similarity_words_ngrams_containment(row["lemmas_0"], row["lemmas_1"], 1, stopwords), axis=1)
    features['2_gram_word_Containment_without_SW_a'] = data.apply(lambda row: similarity_words_ngrams_containment(row["lemmas_0"], row["lemmas_1"], 2, stopwords), axis=1)

    # Word n-gram Containment Similarity Features without Stopwords (b)
    features['1_gram_word_Containment_without_SW_b'] = data.apply(lambda row: similarity_words_ngrams_containment(row["lemmas_1"], row["lemmas_0"], 1, stopwords), axis=1)
    features['2_gram_word_Containment_without_SW_b'] = data.apply(lambda row: similarity_words_ngrams_containment(row["lemmas_1"], row["lemmas_0"], 2, stopwords), axis=1)

    features['average_similarity'] = data.apply(lambda row: average_similarity(row["lemmas_0"], row["lemmas_1"]), axis=1)

    # Lexical Substitution System Feature
    features['lexical_substitution_system'] = data.apply(lambda row: similarity_lemmas(row['lemmas_with_disambiguation_0'], row['lemmas_with_disambiguation_1']), axis=1)


    # ============================
    # Syntactic Features
    # ============================

    features['pathlen_similarity'] = data.apply(lambda row: compute_pathlen_similarity(row["lemmas_0"], row["lemmas_1"], wn), axis=1)
    features['lin_similarity'] = data.apply(lambda row: compute_lin_similarity(row["lemmas_0"], row["lemmas_1"], wn, ic), axis=1)

    # WordNet-Augmented Word Overlap (TakeLab)
    features['wordnet_augmented_overlap'] = data.apply(lambda row: harmonic_mean(
        compute_wordnet_overlap_external([w for w in row["lemmas_0"] if w not in stopwords], [w for w in row["lemmas_1"] if w not in stopwords], wn),
        compute_wordnet_overlap_external([w for w in row["lemmas_1"] if w not in stopwords], [w for w in row["lemmas_0"] if w not in stopwords], wn)
    ), axis=1)

    # Variant WordNet-Augmented Word Overlap (TakeLab)
    features['wordnet_augmented_word_overlap'] = data.apply(lambda row: wordnet_augmented_word_overlap(row["lemmas_0"], row["lemmas_1"], wn), axis=1)

    # Weighted Word Overlap (TakeLab)
    features['weighted_word_overlap'] = data.apply(lambda row: weighted_word_overlap(row["lemmas_0"], row["lemmas_1"], word_freq, total_freq), axis=1)

    # POS tag Ngram Overlap (TakeLab)
    features['pos_2_ngram_overlap'] = data.apply(lambda row: calculate_pos_ngram_overlap(row["lemmas_0"], row["lemmas_1"], nltk, 2), axis=1)
    features['pos_3_ngram_overlap'] = data.apply(lambda row: calculate_pos_ngram_overlap(row["lemmas_0"], row["lemmas_1"], nltk, 3), axis=1)
    features['pos_4_ngram_overlap'] = data.apply(lambda row: calculate_pos_ngram_overlap(row["lemmas_0"], row["lemmas_1"], nltk, 4), axis=1)

    return features
