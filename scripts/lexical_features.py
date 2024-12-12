from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

# +------------------------------------------+
# |     (UKP) Greedy String Tiling           |
# +------------------------------------------+

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
            h, _, _ = karp_rabin_hash(substring, min_length)
            if h not in hashes2:
                hashes2[h] = []
            hashes2[h].append(i)

        # For each t1 substring of length min_length, check if there's a match in t2 using the stored hashes.
        for i in range(n1 - min_length + 1):
            if any(marked1[i:i + min_length]):
                continue
            substring = "".join(t1[i:i + min_length])
            h, _, _ = karp_rabin_hash(substring, min_length)
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


def generate_char_ngrams(tokens, n, nltk):
    merged_string = ''.join(tokens)
    char_ngrams = list(nltk.ngrams(merged_string, n))
    ngram_list = [''.join(gram) for gram in char_ngrams]

    return ' '.join(ngram_list)

def similarity_char_ngrams(tokens_0, tokens_1, n_value):
    ngrams_tokens_0 = generate_char_ngrams(tokens_0, n_value)
    ngrams_tokens_1 = generate_char_ngrams(tokens_1, n_value)

    vectorizer = CountVectorizer().fit([ngrams_tokens_0, ngrams_tokens_1])
    vectors = vectorizer.transform([ngrams_tokens_0, ngrams_tokens_1])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return similarity

def generate_word_ngrams(tokens, n, nltk):
    word_ngrams = list(nltk.ngrams(tokens, n))
    return set(word_ngrams)

def compute_containment(set_a, set_b):
    intersection = set_a.intersection(set_b)
    containment = len(intersection) / len(set_a)
    return containment

def similarity_words_ngrams_jaccard(tokens_0, tokens_1, n_value, nltk, stopwords=None):

    if stopwords is not None:
      tokens_0 = [token for token in tokens_0 if token not in stopwords]
      tokens_1 = [token for token in tokens_1 if token not in stopwords]

    if len(tokens_0) == 0 or len(tokens_1) == 0 or (len(tokens_0) < n_value and len(tokens_1) < n_value):
      return 0.0

    ngrams_tokens_0 = generate_word_ngrams(tokens_0, n_value, nltk)
    ngrams_tokens_1 = generate_word_ngrams(tokens_1, n_value, nltk)

    similarity = 1 - nltk.metrics.distance.jaccard_distance(ngrams_tokens_0, ngrams_tokens_1)

    return similarity

def similarity_words_ngrams_containment(tokens_0, tokens_1, n_value, stopwords=None):

    if stopwords is not None:
      tokens_0 = [token for token in tokens_0 if token not in stopwords]
      tokens_1 = [token for token in tokens_1 if token not in stopwords]

    if len(tokens_0) == 0 or len(tokens_1) == 0:
      return 0.0

    ngrams_tokens_0 = generate_word_ngrams(tokens_0, n_value)
    ngrams_tokens_1 = generate_word_ngrams(tokens_1, n_value)

    if len(ngrams_tokens_0) == 0:
      return 0.0
    similarity = compute_containment(ngrams_tokens_0, ngrams_tokens_1)

    return similarity


# +------------------------------------------+
# | (UKP) WordNet's Wu-Palmer similarity     |
# +------------------------------------------+

def calculate_similarity(word_1, word_2, wn):
    word1_synsets = wn.synsets(word_1)
    word2_synsets = wn.synsets(word_2)
    highest_similarity = 0.0
    for syn1 in word1_synsets:
        for syn2 in word2_synsets:
            similarity_score = syn1.wup_similarity(syn2)
            if similarity_score and similarity_score > highest_similarity:
                highest_similarity = similarity_score
    return highest_similarity

def average_similarity(tokens_0, tokens_1):
    total_similarity = 0.0
    word_count = 0
    for token_0 in tokens_0:
        best_match_similarity = 0.0
        for token_1 in tokens_1:
            similarity = calculate_similarity(token_0, token_1)
            if similarity and similarity > best_match_similarity:
                best_match_similarity = similarity
        total_similarity += best_match_similarity
        word_count += 1
    return total_similarity / word_count if word_count > 0 else 0.0

# +------------------------------------------+
# | (UKP) Simmilarity Function               |
# +------------------------------------------+

def similarity_lemmas(tokens_0, tokens_1):

    text_0 = ' '.join(tokens_0)
    text_1 = ' '.join(tokens_1)

    vectorizer = CountVectorizer().fit([text_0, text_1])
    vectors = vectorizer.transform([text_0, text_1])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return similarity