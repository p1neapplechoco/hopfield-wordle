from .ModernHopfieldNetwork import ModernHopfieldNetwork as hf

import string
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter, defaultdict

# Constants for encoding
ALPHABET = list(string.ascii_lowercase)
LETTER_DIM = len(ALPHABET)
WORD_LEN = 5
D = WORD_LEN * LETTER_DIM


def encode_word(word: str) -> np.ndarray:
    """
    One-hot encode a 5-letter word into a vector of length D.

    Args:
        word: A lowercase string of length 5 containing letters a-z.

    Returns:
        A numpy array of shape (D,) representing the one-hot encoding.
    """
    pattern = np.zeros(D, dtype=float)
    word = word.lower()
    for pos, c in enumerate(word):
        if c in ALPHABET:
            idx = ALPHABET.index(c)
            pattern[pos * LETTER_DIM + idx] = 1.0
    return pattern


def anchor_pattern(pattern: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an initial state and anchor mask from a pattern with wildcards.

    A '?' in the pattern represents an unknown letter. Known letters
    are encoded as one-hot in state; anchor masks are 1.0 for each
    letter position that is fixed and 0.0 otherwise.

    Args:
        pattern: A 5-character string containing letters a-z or '?'.

    Returns:
        state: one-hot array of length D with 1.0 at known-letter positions.
        anchor: mask array of length D indicating fixed positions.
    """
    state = np.zeros(D, dtype=float)
    anchor = np.zeros(D, dtype=float)
    pattern = pattern.lower()

    for pos, c in enumerate(pattern):
        if c in ALPHABET:
            idx = ALPHABET.index(c)
            base = pos * LETTER_DIM
            state[base + idx] = 1.0
            anchor[base : base + LETTER_DIM] = 1.0

    return state, anchor


class WordleSolver:
    """
    A Wordle solver using a modern Hopfield network for retrieval.

    Maintains position constraints (greens/yellows) and global letter counts.
    Filters the dictionary on each guess and scores candidates via Hopfield.
    """

    def __init__(self, words: List[str], beta: float = 1.0) -> None:
        """
        Initialize solver with a list of valid words.

        Args:
            words: Full list of candidate 5-letter words.
            beta: Inverse-temperature parameter for Hopfield network.
        """
        self.all_words = words.copy()
        self.beta = beta
        self.reset()

    def reset(self) -> None:
        """
        Reset constraints and restore the full candidate list.
        """
        self.pos_require: List[Optional[str]] = [None] * WORD_LEN
        self.pos_exclude: List[set] = [set() for _ in range(WORD_LEN)]
        self.min_count: defaultdict = defaultdict(int)
        self.max_count: dict = {c: WORD_LEN for c in ALPHABET}
        self.words: List[str] = self.all_words.copy()

    def update_constraint(self, guess: str, result: np.ndarray) -> None:
        """
        Update position and count constraints based on Wordle feedback.

        Feedback codes per position:
            2: green (correct letter and position)
            1: yellow (correct letter, wrong position)
            0: gray (absent or over-count)

        Args:
            guess: The guessed word.
            result: Array of length 5 with values in {{0,1,2}}.
        """
        total_counts = Counter(guess)
        confirmed = Counter()

        # Greens: fixed letters
        for i, (g, r) in enumerate(zip(guess, result)):
            if r == 2:
                self.pos_require[i] = g
                confirmed[g] += 1
                self.min_count[g] = max(self.min_count[g], confirmed[g])

        # Yellows: letter present but position excluded
        for i, (g, r) in enumerate(zip(guess, result)):
            if r == 1:
                self.pos_exclude[i].add(g)
                confirmed[g] += 1
                self.min_count[g] = max(self.min_count[g], confirmed[g])

        # Grays: adjust max counts
        for letter, occ in total_counts.items():
            conf = confirmed[letter]
            # if no confirmations and at least one gray, letter banned
            if conf == 0 and any(
                guess[i] == letter and result[i] == 0 for i in range(WORD_LEN)
            ):
                self.max_count[letter] = 0
            # if over-guessed, cap at confirmed count
            elif occ > conf:
                self.max_count[letter] = conf

    def update_words(self) -> None:
        """
        Re-filter candidate list based on current constraints.
        """

        def valid(w: str) -> bool:
            # Fixed positions
            for i, req in enumerate(self.pos_require):
                if req and w[i] != req:
                    return False
            # Excluded letters
            for i, bans in enumerate(self.pos_exclude):
                if w[i] in bans:
                    return False
            # Global frequency constraints
            wc = Counter(w)
            for c, mn in self.min_count.items():
                if wc[c] < mn:
                    return False
            for c, mx in self.max_count.items():
                if wc[c] > mx:
                    return False
            return True

        self.words = [w for w in self.all_words if valid(w)]

    def possible_answers(
        self, pattern: str, top_k: int = 5
    ) -> Tuple[List[str], np.ndarray]:
        """
        Suggest top candidate words matching a pattern.

        Args:
            pattern: String of length 5 with letters or '?' for unknowns.
            top_k: Number of top-scoring candidates to return.

        Returns:
            A tuple (candidates, scores):
              - candidates: List of words (max length top_k).
              - scores: Numpy array of their Hopfield scores.
        """
        # Apply filters
        self.update_words()
        n = len(self.words)
        print(f"{n} candidates remain after filtering.")

        if n == 0:
            print("No matches found. Consider resetting or revising constraints.")
            return [], np.array([])

        # Build and run Hopfield network
        network = hf(beta=self.beta, verbose=False)
        for w in self.words:
            network.add_pattern(encode_word(w))

        init_state, anchor = anchor_pattern(pattern)
        idxs, scores = network.retrieve_candidates(
            init_state=init_state, anchor=anchor, max_iter=5, top_k=top_k
        )

        candidates = [self.words[i] for i in idxs]
        return candidates, scores
