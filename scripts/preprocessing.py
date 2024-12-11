import string
import nltk

wnl = nltk.stem.WordNetLemmatizer()

def replace_contractions(tokens):
  """
  Replace specified contractions in a list of tokens, as specified by TakeLab.

  Parameters:
    tokens (list): The list of tokens to process.

  Returns:
    list: The list of tokens after replacing the specified contractions.
  """
  contractions_dict = {
      "n’t": "not", "n't": "not",
      "’m": "am", "'m": "am",
      "’ve": "have", "'ve": "have",
      "’re": "are", "'re": "are",
      "’ll": "will", "'ll": "will",
  }
  
  new_tokens = []
  for t in tokens:
      lower_t = t.lower()
      if lower_t in contractions_dict:
          new_tokens.append(contractions_dict[lower_t])
      else:
          new_tokens.append(t)
  return new_tokens


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