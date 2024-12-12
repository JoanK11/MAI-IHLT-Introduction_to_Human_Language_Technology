import numpy as np
from collections import Counter
from statistics import harmonic_mean
from itertools import product

def compute_pathlen_similarity(list1, list2, wn):
    pathlen_scores = []

    for word1 in list1:
        for word2 in list2:
            synsets1 = wn.synsets(word1)
            synsets2 = wn.synsets(word2)

            if synsets1 and synsets2:
                scores = [
                    s1.path_similarity(s2) for s1, s2 in product(synsets1, synsets2) if s1.path_similarity(s2) is not None
                ]
                if scores:
                    pathlen_scores.append(max(scores))

    if pathlen_scores:
        average_similarity = sum(pathlen_scores) / len(pathlen_scores)
    else:
        average_similarity = 0

    return average_similarity

def compute_lin_similarity(list1, list2, wn, ic):
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
                    except Exception:
                        continue
                if scores:
                    lin_scores.append(max(scores))

    if lin_scores:
        average_similarity = sum(lin_scores) / len(lin_scores)
    else:
        average_similarity = 0

    return average_similarity


# +-----------------------------------------------------------------+
# | (TakeLab) WordNet-Augmented Word Overlap External Harmonic Mean |
# +-----------------------------------------------------------------+

def compute_wordnet_overlap_external(tokens_0, tokens_1, wn):

    if not tokens_0:
        return 0.0

    total_similarity_score = 0.0
    comparison_set = set(tokens_1)

    for token in tokens_0:
        if token in comparison_set:
            total_similarity_score += 1.0
        else:
            best_match_score = 0.0
            for candidate in tokens_1:
                similarity_val = wordnet_path_similarity(token, candidate, wn)
                if similarity_val > best_match_score:
                    best_match_score = similarity_val
            total_similarity_score += best_match_score

    # Compute the average similarity across all items in the first list
    return total_similarity_score / len(tokens_0)

def wordnet_path_similarity(word1, word2, wn):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    # Consider only pairs with matching POS, and take the maximum similarity
    max_sim = 0.0
    for s1 in synsets1:
        for s2 in synsets2:
            if s1.pos() == s2.pos():
                sim = s1.path_similarity(s2)
                if sim is not None and sim > max_sim:
                    max_sim = sim
    return max_sim


# +--------------------------------------------------+
# | (TakeLab) WordNet-Augmented Word Overlap         |
# +--------------------------------------------------+

def score(word, sentence, wn):
    if word in sentence:
        return 1
    return max(wordnet_path_similarity(word, w, wn) for w in sentence if wordnet_path_similarity(word, w, wn) is not None) or 0

def pwn(sentence1, sentence2, wn):
    total_score = sum(score(word, sentence2, wn) for word in sentence1)
    return total_score / len(sentence2) if len(sentence2) > 0 else 0


def wordnet_augmented_word_overlap(sentence1, sentence2, wn):
    pwn_s1_s2 = pwn(sentence1, sentence2, wn)
    pwn_s2_s1 = pwn(sentence2, sentence1, wn)
    
    score = harmonic_mean(pwn_s1_s2, pwn_s2_s1)

    return score

# +----------------------------------+
# | (TakeLab) Weighted Word Overlap  |
# +----------------------------------+

def information_content(w, word_freq, total_freq):
    freq = word_freq[w.lower()] if w.lower() in word_freq else 1
    ic = np.log(total_freq / freq)
    return ic

def weighted_word_overlap(S1, S2, word_freq, total_freq):
    ic_intersect = sum(information_content(w, word_freq, total_freq) for w in set(S1).intersection(set(S2)))
    ic_total = sum(information_content(w, word_freq, total_freq) for w in set(S1 + S2))
    return (2 * ic_intersect / ic_total) if ic_total > 0 else 0.0

# +----------------------------------+
# | (TakeLab) POS Tag Ngram Overlap  |
# +----------------------------------+

def calculate_pos_ngram_overlap(lemmas_0, lemmas_1, nltk, n=2):

    pos_ngrams0 = list(nltk.util.ngrams(lemmas_0, n))
    pos_ngrams1 = list(nltk.util.ngrams(lemmas_1, n))

    ngrams0_counts = Counter(pos_ngrams0)
    ngrams1_counts = Counter(pos_ngrams1)
    
    intersection = sum((ngrams1_counts & ngrams1_counts).values())
    total_0 = sum(ngrams0_counts.values())
    total_1 = sum(ngrams1_counts.values())
    
    if total_0 == 0 or total_1 == 0:
        return 0

    overlap = harmonic_mean([intersection / total_0, intersection / total_1])
    return overlap