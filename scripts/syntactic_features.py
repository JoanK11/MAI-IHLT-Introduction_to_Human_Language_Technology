import numpy as np
from collections import Counter
from statistics import harmonic_mean
from itertools import product

def compute_pathlen_similarity(list1, list2, wn):
    """
    Compute Path Length similarity scores between two lists of words using WordNet
    and return the similarity scores for all pairs and their average.
    """
    pathlen_scores = []

    # Iterate through all word pairs
    for word1 in list1:
        for word2 in list2:
            # Get synsets for each word
            synsets1 = wn.synsets(word1)
            synsets2 = wn.synsets(word2)

            if synsets1 and synsets2:  # Ensure both words have synsets
                # Compute Path Length Similarity for all synset pairs
                scores = [
                    s1.path_similarity(s2) for s1, s2 in product(synsets1, synsets2) if s1.path_similarity(s2) is not None
                ]
                # Append the maximum score to the list if any scores exist
                if scores:
                    pathlen_scores.append(max(scores))

    # Compute the average similarity
    if pathlen_scores:
        average_similarity = sum(pathlen_scores) / len(pathlen_scores)
    else:
        average_similarity = 0  # No valid similarities found

    return average_similarity

def compute_lin_similarity(list1, list2, wn, ic):
    """
    Compute Lin similarity scores between two lists of words using WordNet
    and return the similarity scores for all pairs and their average.
    """
    lin_scores = []

    # Iterate through all word pairs
    for word1 in list1:
        for word2 in list2:
            # Get synsets for each word
            synsets1 = wn.synsets(word1)
            synsets2 = wn.synsets(word2)

            if synsets1 and synsets2:  # Ensure both words have synsets
                # Compute Lin Similarity for all synset pairs
                scores = []
                for s1, s2 in product(synsets1, synsets2):
                    try:
                        similarity = s1.lin_similarity(s2, ic)
                        if similarity is not None:
                            scores.append(similarity)
                    except Exception:
                        continue  # Skip if there's an error in calculation
                # Append the maximum score to the list if any scores exist
                if scores:
                    lin_scores.append(max(scores))

    # Compute the average similarity
    if lin_scores:
        average_similarity = sum(lin_scores) / len(lin_scores)
    else:
        average_similarity = 0  # No valid similarities found

    return average_similarity


# +-----------------------------------------------------------------+
# | (TakeLab) WordNet-Augmented Word Overlap External Harmonic Mean |
# +-----------------------------------------------------------------+

def P_WN(S1, S2, wn):
    """
    Compute P_WN(S1, S2) metric as described in the TakeLab paper.

    Parameters:
        S1 (list): List of tokenized words from the first sentence.
        S2 (list): List of tokenized words from the second sentence.

    Returns:
        float: The computed P_WN(S1, S2) value.
    """
    if len(S1) == 0:
        return 0.0

    score = 0.0
    S2_set = set(S2) # Slight optimization for membership checks
    for word1 in S1:
        if word1 in S2_set:
            score += 1.0
        else:
            # Find the best similarity if exact match is not found
            best_sim = max((wordnet_path_similarity(word1, word2, wn) for word2 in S2), default=0.0)
            score += best_sim

    return score / len(S1)

def wordnet_path_similarity(word1, word2, wn):
    """
    Compute the maximum WordNet path similarity between all synset pairs of two given words.
    Only consider synsets that share the same part-of-speech (POS).

    Parameters:
        word1 (str): First word.
        word2 (str): Second word.

    Returns:
        float: Maximum path similarity between word1 and word2.
    """
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

# Compute score(w, S) as defined in the formula
def score(word, sentence, wn):
    if word in sentence:
        return 1
    return max(wordnet_path_similarity(word, w, wn) for w in sentence if wordnet_path_similarity(word, w, wn) is not None) or 0

# Compute PWN(S1, S2)
def pwn(sentence1, sentence2, wn):
    total_score = sum(score(word, sentence2, wn) for word in sentence1)
    return total_score / len(sentence2) if len(sentence2) > 0 else 0

# WordNet-Augmented Word Overlap
def wordnet_augmented_word_overlap(sentence1, sentence2, wn):
    # Compute PWN(S1, S2) and PWN(S2, S1)
    pwn_s1_s2 = pwn(sentence1, sentence2, wn)
    pwn_s2_s1 = pwn(sentence2, sentence1, wn)
    
    score = harmonic_mean(pwn_s1_s2, pwn_s2_s1)

    return score

# +----------------------------------+
# | (TakeLab) Weighted Word Overlap  |
# +----------------------------------+

def information_content(w, word_freq, total_freq):
    """
    Compute the information content of a word.

    Args:
        w (str): Word.

    Returns:
        float: Information content.
    """
    freq = word_freq[w.lower()] if w.lower() in word_freq else 1
    ic = np.log(total_freq / freq)
    return ic

def weighted_word_overlap(S1, S2, word_freq, total_freq):
    """
    Compute the weighted word overlap between two sentences.

    Args:
        S1 (list): List of words from sentence 1.
        S2 (list): List of words from sentence 2.

    Returns:
        float: Weighted word overlap.
    """
    ic_intersect = sum(information_content(w, word_freq, total_freq) for w in set(S1).intersection(set(S2)))
    ic_total = sum(information_content(w, word_freq, total_freq) for w in set(S1 + S2))
    return (2 * ic_intersect / ic_total) if ic_total > 0 else 0.0

# +----------------------------------+
# | (TakeLab) POS Tag Ngram Overlap  |
# +----------------------------------+

def calculate_pos_ngram_overlap(lemmas_0, lemmas_1, nltk, n=2):
    """
    Calculate POS-tagged ngram overlap between two texts.
    Args:
    - text1, text2: Input strings for comparison.
    - n: The size of ngram (e.g., 1 for unigram, 2 for bigram).
    
    Returns:
    - overlap: The POS ngram overlap score.
    """
    
    # Create ngrams of POS tags
    pos_ngrams0 = list(nltk.util.ngrams(lemmas_0, n))
    pos_ngrams1 = list(nltk.util.ngrams(lemmas_1, n))
    
    # Count ngrams
    ngrams0_counts = Counter(pos_ngrams0)
    ngrams1_counts = Counter(pos_ngrams1)
    
    # Calculate overlap
    intersection = sum((ngrams1_counts & ngrams1_counts).values())
    total_0 = sum(ngrams0_counts.values())
    total_1 = sum(ngrams1_counts.values())
    
    if total_0 == 0 or total_1 == 0:
        return 0  # Avoid division by zero
    
    # Harmonic mean
    overlap = harmonic_mean([intersection / total_0, intersection / total_1])
    return overlap