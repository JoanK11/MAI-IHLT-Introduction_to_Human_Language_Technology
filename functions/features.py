import numpy as np
from nltk.corpus import wordnet as wn

# +------------------------------------------+
# |      (UKP) Longest Common Substring      |
# +------------------------------------------+

def longest_common_substring(sentence_0, sentence_1):
    """
    Compute the length of the longest common substring between two sequences of tokens.

    Parameters:
        sentence_0 (list): List of tokens from the first sentence.
        sentence_1 (list): List of tokens from the second sentence.

    Returns:
        int: The length of the longest common substring found.
    """
    m, n = len(sentence_0), len(sentence_1)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    longest_len = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if sentence_0[i - 1] == sentence_1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest_len:
                    longest_len = dp[i][j]
            else:
                dp[i][j] = 0

    return longest_len

# +------------------------------------------+
# |     (UKP) Longest Common Subsequence     |
# +------------------------------------------+

def longest_common_subsequence(sentence_0, sentence_1):
    """
    Compute the length of the longest common subsequence (LCS) between two sequences of tokens.

    Parameters:
        sentence_0 (list): List of tokens from the first sentence.
        sentence_1 (list): List of tokens from the second sentence.

    Returns:
        int: The length of the longest common subsequence found.
    """
    m, n = len(sentence_0), len(sentence_1)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if sentence_0[i - 1] == sentence_1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    return lcs_length

def optimized_gst(tokens1, tokens2, min_match_length=3):
    """
    Compute the longest matched substring between two token sequences using a Greedy String Tiling (GST) approach.

    Parameters:
        tokens1 (list): List of tokens from the first sequence.
        tokens2 (list): List of tokens from the second sequence.
        min_match_length (int): The minimum length of a match to be considered as a tile.

    Returns:
        int: The length of the longest tile (contiguous match) found between the two token sequences.
    """

    def karp_rabin_hash(s, length):
        # Compute the rolling hash for a substring s of given length using base/mod.
        base, mod = 257, 10**9 + 7
        h = 0
        for i in range(length):
            h = (h * base + ord(s[i])) % mod
        return h, base, mod

    def find_maximal_matches(t1, t2, min_length, marked1, marked2):
        # Find all maximal matches of at least min_length length that are not yet marked.
        matches = []
        n1, n2 = len(t1), len(t2)
        hashes2 = {}

        # Precompute hashes for t2 substrings of length min_length and store their start indices.
        for i in range(n2 - min_length + 1):
            if any(marked2[i:i + min_length]):
                continue
            substring = "".join(t2[i:i + min_length])
            h, base, mod = karp_rabin_hash(substring, min_length)
            if h not in hashes2:
                hashes2[h] = []
            hashes2[h].append(i)

        # For each t1 substring of length min_length, check if there's a match in t2 using the stored hashes.
        for i in range(n1 - min_length + 1):
            if any(marked1[i:i + min_length]):
                continue
            substring = "".join(t1[i:i + min_length])
            h, base, mod = karp_rabin_hash(substring, min_length)
            if h in hashes2:
                for j in hashes2[h]:
                    match_length = 0
                    # Extend the match as long as tokens continue to match and are unmarked.
                    while (
                        i + match_length < n1 and
                        j + match_length < n2 and
                        t1[i + match_length] == t2[j + match_length] and
                        not marked1[i + match_length] and
                        not marked2[j + match_length]
                    ):
                        match_length += 1
                    if match_length >= min_length:
                        matches.append((i, j, match_length))

        # Sort matches by decreasing length so that we pick the longest matches first.
        return sorted(matches, key=lambda x: -x[2])

    def mark_tiles(matches, marked1, marked2):
        # Mark the tokens covered by the newly found tiles so they won't be reused.
        tiles = []
        for i, j, length in matches:
            if not any(marked1[i:i + length]) and not any(marked2[j:j + length]):
                tiles.append((i, j, length))
                for k in range(length):
                    marked1[i + k] = True
                    marked2[j + k] = True
        return tiles

    marked1 = [False] * len(tokens1)
    marked2 = [False] * len(tokens2)
    tiles = []

    # Repeatedly find matches and mark them until no more can be found.
    while True:
        matches = find_maximal_matches(tokens1, tokens2, min_match_length, marked1, marked2)
        if not matches:
            break
        new_tiles = mark_tiles(matches, marked1, marked2)
        tiles.extend(new_tiles)
        # Decrease min_match_length to try finding shorter matches after the longest ones are found.
        if min_match_length > 1:
            min_match_length -= 1

    # Return the length of the longest tile found.
    max_tile_length = max((tile[2] for tile in tiles), default=0)

    return max_tile_length

# +------------------------------------------+
# | (TakeLab) WordNet-Augmented Word Overlap |
# +------------------------------------------+

def P_WN(S1, S2):
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
            best_sim = max((wordnet_path_similarity(word1, word2) for word2 in S2), default=0.0)
            score += best_sim

    return score / len(S1)

def wordnet_path_similarity(word1, word2):
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

def harmonic_mean2(x, y):
    """
    Compute the harmonic mean of two numbers.

    Parameters:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The harmonic mean of the two numbers.
    """
    if (x + y) > 0:
        return 2 * x * y / (x + y)
    else:
        return 0