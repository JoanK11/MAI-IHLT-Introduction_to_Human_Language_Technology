from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# +------------------------------------------+
# |      (UKP) Longest Common Substring      |
# +------------------------------------------+

def longest_common_substring(sentence_0, sentence_1):

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

def greedy_string_tiling(tokens1, tokens2, min_match_length=3):

    matched = []
    used1 = [False] * len(tokens1)
    used2 = [False] * len(tokens2)
    
    while True:
        max_match_length = 0
        max_match = None
        
        for i in range(len(tokens1)):
            if used1[i]:
                continue
            for j in range(len(tokens2)):
                if used2[j]:
                    continue
                
                length = 0
                while (i + length < len(tokens1) and j + length < len(tokens2) and
                       tokens1[i + length] == tokens2[j + length] and
                       not used1[i + length] and not used2[j + length]):
                    length += 1
                
                if length >= min_match_length and length > max_match_length:
                    max_match_length = length
                    max_match = (i, j, length)
        
        if max_match is None:
            break
        
        i, j, length = max_match
        for k in range(length):
            used1[i + k] = True
            used2[j + k] = True
        matched.append((i, j, length))
    
    return len(matched)


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