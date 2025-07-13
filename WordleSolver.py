from ModernHopfieldNetwork import ModernHopfieldNetwork as hf
import string
import numpy as np
from typing import List, Tuple

ALPHABET = list(string.ascii_lowercase)
LETTER_DIM = len(ALPHABET)
WORD_LEN = 5
D = WORD_LEN * LETTER_DIM


def encode_word(word):
    """
    One-hot encoder.
    """
    pattern = np.zeros(D, dtype=float)

    for pos, c in enumerate(word.lower()):
        idx = ALPHABET.index(c)
        pattern[pos * LETTER_DIM + idx] = 1.0

    return pattern


def decode_word(pattern: np.ndarray):
    """
    One-hot decoder.
    """
    pattern = pattern.reshape(WORD_LEN, LETTER_DIM)
    return "".join(ALPHABET[int(np.argmax(pattern[pos]))] for pos in range(WORD_LEN))


def anchor_pattern(pattern: str):
    """
    Given a 5-letter pattern with '?' as wildcards,
    returns (state, anchor) both shape (130,):
      - state:  one-hot encoding for known letters, zeros for '?' slots
      - anchor :  1.0 for all entries in known-letter blocks, 0.0 for '?' blocks
    """
    state = np.zeros(D, dtype=float)
    anchor = np.zeros(D, dtype=float)

    for pos, c in enumerate(pattern.lower()):
        start = pos * LETTER_DIM
        end = start + LETTER_DIM

        if c in ALPHABET:
            idx = ALPHABET.index(c)
            state[start + idx] = 1.0
            anchor[start:end] = 1.0

    return state, anchor


class WordleWSolver:
    def __init__(self, words: List, beta: float = 1.0):
        self.words = words.copy()

        self.include = set()
        self.exclude = set()

        self.beta = beta

    def add_word(self, word):
        self.words.append(word)

    def add_constraint(self, include: List, exclude: List):
        for c in include:
            self.include.add(c)

        for c in exclude:
            self.exclude.add(c)

    def update_words(self):
        if len(self.include) > 5 or len(self.exclude) == LETTER_DIM:
            self.words = list()

        new_words = list()

        for word in self.words:
            to_include = len(set(self.include) - set(list(word))) == 0
            to_exclude = set(list(word)) - set(self.exclude) != set(list(word))

            if to_exclude or not to_include:
                continue

            new_words.append(word)

        self.words = new_words.copy()

    def possible_answers(
        self, word_to_solve: str, top_k: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Given a word_to_solve with "?" as wildcards, returns top_k candidates for that word
        """

        self.update_words()
        print(f"Current included characters: {self.include}")
        print(f"Current excluded characters: {self.exclude}")
        print(f"Number of words remaining: {len(self.words)}")

        solver = hf(beta=self.beta, verbose=False)

        patterns = np.stack([encode_word(word) for word in self.words])

        for pattern in patterns:
            solver.add_pattern(pattern=pattern)

        pattern, anchor = anchor_pattern(pattern=word_to_solve)

        idxs, scores = solver.retrieve_candidates(
            init_state=pattern, anchor=anchor, max_iter=5, top_k=top_k
        )

        words = [self.words[i] for i in idxs]
        return words, scores
