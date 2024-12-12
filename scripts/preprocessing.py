# Function to lemmatize based on POS tagging
def lemmatize_WNL(p, wnl, warn=False):
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
def lemmatize(row, name, nltk, wnl, warn=False):
    tokens = row[name]
    tokens = [token.lower() for token in tokens]
    pair_tags = nltk.pos_tag(tokens)
    return [lemmatize_WNL(pair, wnl, warn) for pair in pair_tags]

def map_pos_to_wordnet(pos):
  if pos.startswith('N'):
      return 'n'
  elif pos.startswith('V'):
      return 'v'
  elif pos.startswith('J'):
      return 'a'
  elif pos.startswith('R'):
      return 'r'
  else:
      return None

def tokens_to_synsets_name(tokens, nltk):
  pairs = nltk.tag.pos_tag(tokens)
  mapped_pairs = [(word, map_pos_to_wordnet(pos)) for word, pos in pairs]

  tokens_with_synsets = [pair[0] for pair in mapped_pairs if pair[1] is None] #Keep non open words

  possible_synsets = [pair for pair in mapped_pairs if pair[1] is not None]
  tokens_filtered = [pair[0] for pair in possible_synsets]
  POS = [pair[1] for pair in possible_synsets]

  for token, pos in zip(tokens_filtered, POS):
    synset = nltk.wsd.lesk(tokens, token, pos)

    if not synset:  # If no synset found (e.g., POS tagger failure example: happens with "fervent" POS "a"))
        # Use the most frequent synset if available
        synset = nltk.corpus.wordnet.synsets(token)
        synset = synset[0] if synset else None

    # Add synset name if available
    if synset:
        tokens_with_synsets.append(synset.name())

  return tokens_with_synsets