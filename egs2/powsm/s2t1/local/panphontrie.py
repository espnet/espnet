import pickle
import string
import unicodedata


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.value = None

    def insert(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end_of_word = True
        node.value = word

    def serialize(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def deserialize(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


def build_trie_from_file(input_path, output_pickle):
    """
    List of panphon phone entries (one per line) -> trie
    input_path: path to file of panphon phone entries
    (first column of https://github.com/dmort27/panphon/blob/master/panphon/data/ipa_all.csv)
    output_pickle: path to the output trie object serialized as pickle file
    """
    root = TrieNode()
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                root.insert(word)
    root.insert(" ")  # also insert space
    root.serialize(output_pickle)
    return root


def load_trie(pickle_path):
    """
    Load trie object from pickle file
    pickle_path: path to pickle file of trie object
    return: root node of the trie
    """
    return TrieNode.deserialize(pickle_path)


removepunc = str.maketrans("", "", string.punctuation)
customized = {"ʡ": "ʔ", "ᶑ": "ɗ", "g": "ɡ"}
supraseg = {"ː", "ˑ", "̆", "͜"}


def clean(sequence, with_supraseg=True):
    """
    Normalize phones' unicode so that trie search can handle everything
    Remove suprasegmental diacritics if specified
    """
    sequence = unicodedata.normalize("NFD", sequence)
    sequence = sequence.translate(removepunc)
    sequence = "".join([customized.get(c, c) for c in sequence])
    if not with_supraseg:
        sequence = "".join([c for c in sequence if c not in supraseg])
    return sequence


def panphon_gsearch(seq, root, with_supraseg=True):
    """
    Greedy longest-match-first search in panphon trie.
    seq: input sequence (string)
    root: root node of the trie
    with_supraseg: if False, remove suprasegmental diacritics before search
    return: list of phones and set of OOV phones
    """
    seq = clean(seq, with_supraseg)  # fix unicode and remove punctuations
    res, oov = [], set()
    i, N = 0, len(seq)
    while i < N:
        node = root
        start = i
        last_value = None
        last_match = i
        # Search for the longest match
        while i < N and seq[i] in node.children:
            node = node.children[seq[i]]
            i += 1
            if node.is_end_of_word:
                last_value = node.value
                last_match = i
        if last_value is not None:
            res.append(last_value)
        # Deal with possibly trailing diacritics of OOV phone
        while i < N and seq[i] not in root.children:
            i += 1
        if i != last_match:
            oov.add((seq[start:i], last_value))

    return res, oov
