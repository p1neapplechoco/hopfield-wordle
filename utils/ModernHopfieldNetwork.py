import numpy as np
from scipy.special import softmax
from typing import List, Tuple


class ModernHopfieldNetwork:
    def __init__(self, beta: float = 1.0, verbose: bool = True):
        self.beta = beta
        self.verbose = verbose
        self.patterns = []
        self.d = None

    def add_pattern(self, pattern: np.ndarray):
        """
        Adds a pattern to the network
        """
        if not isinstance(pattern, np.ndarray):
            raise ValueError("Pattern must be a numpy array.")

        if self.d is None:
            self.d = pattern.size

        if pattern.size != self.d:
            raise ValueError("All patterns should be the same size.")

        self.patterns.append(pattern)

        if self.verbose:
            print(
                f"Added pattern: {pattern}, current total patterns: {len(self.patterns)}"
            )

    def update_patterns(self, patterns: np.ndarray):
        self.patterns = patterns
        self.d = self.patterns[0].size

    def __update_state(self, state: np.ndarray) -> np.ndarray:
        sims = np.array(self.patterns) @ state
        activation = softmax(self.beta * sims)
        updated_state = np.array(self.patterns).T @ activation
        return updated_state

    def __settle(
        self, init_state: np.ndarray, anchor: np.ndarray, max_iter: int
    ) -> np.ndarray:
        state = init_state.copy()

        for it in range(max_iter):
            updated = self.__update_state(state)
            new_state = updated * (1 - anchor) + init_state * anchor
            state = new_state.copy()

        return state

    def predict(
        self,
        init_state: np.ndarray,
        anchor: np.ndarray = None,
        max_iter: int = 10,
    ) -> np.ndarray:
        """
        Returns the best matching candidate from stored patterns.
        """
        if anchor is None:
            anchor = np.zeros_like(init_state, dtype=float)

        state = self.__settle(init_state=init_state, anchor=anchor, max_iter=max_iter)

        sims = np.array(self.patterns) @ state
        idx = int(np.argmax(sims))
        final = np.array(self.patterns)[idx]
        return final * (1.0 - anchor) + init_state * anchor

    def retrieve_candidates(
        self,
        init_state: np.ndarray,
        anchor: np.ndarray = None,
        max_iter: int = 10,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the candidates and their corresponding score from stored patterns.
        """
        if anchor is None:
            anchor = np.zeros_like(init_state, dtype=float)

        state = self.__settle(init_state=init_state, anchor=anchor, max_iter=max_iter)

        sims = np.array(self.patterns) @ state

        anchored_positions = anchor.astype(bool)
        if anchored_positions.any():
            valid = np.all(
                np.array(self.patterns)[:, anchored_positions]
                == init_state[anchored_positions],
                axis=1,
            )
            sims = np.where(valid, sims, -np.inf)

        N = sims.shape[0]
        if top_k < N:
            # get unsorted top_k indices
            topk_unsorted = np.argpartition(sims, -top_k)[-top_k:]
            # sort those top_k
            idxs = topk_unsorted[np.argsort(sims[topk_unsorted])[::-1]]
        else:
            idxs = np.argsort(sims)[::-1]

        return idxs, sims[idxs]
