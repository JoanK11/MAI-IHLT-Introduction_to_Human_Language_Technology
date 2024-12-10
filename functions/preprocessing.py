import string
import nltk

wnl = nltk.stem.WordNetLemmatizer()

def remove_punctuation_tokens(tokens):
    return [token for token in tokens if token not in string.punctuation]

# Function to lemmatize based on POS tagging
def lemmatize_WNL(p, warn=False):
  d = {'NN': 'n', 'NNS': 'n',
       'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
       'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
       'RB': 'r', 'RBR': 'r', 'RBS': 'r'}
  if p[1] in d:
    return wnl.lemmatize(p[0], pos=d[p[1]])
  else:
    """
    if warn:
      warnings.warn(f"Unrecognized POS tag '{p[1]}' for word '{p[0]}'.", category=UserWarning)
    """
    return p[0]

# Function to POS tag and lemmatize tokens in a sentence
def lemmatize(row, name, warn=False):
  tokens = row[name]
  tokens = [token.lower() for token in tokens]
  pair_tags = nltk.pos_tag(tokens)
  return [lemmatize_WNL(pair, warn) for pair in pair_tags]