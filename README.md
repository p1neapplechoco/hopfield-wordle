# Wordle-Hopfield Solver

A proof-of-concept Wordle solver that uses a modern Hopfield network to retrieve the most plausible answers given partial feedback.

## Features

- **Neural pattern retrieval**  
  Uses “Hopfield Network Is All You Need” (Ramsauer et al., 2020) for fast associative recall of 5-letter words.

- **Constraint propagation**  
  Enforces Wordle feedback (green/yellow/gray) to prune the dictionary before network retrieval.

- **Top-K ranking**  
  Returns the K highest-score candidates at each step.


## Intuition

This solver relies on the principle of **associative memory**:

- **Pattern completion**  
  A Hopfield network stores each 5-letter word as a distinct attractor in its energy landscape. When you feed in a partial or “noisy” pattern (with some letters fixed and others blank), the network naturally relaxes toward the nearest stored word.

- **One-hot encoding**  
  We represent every letter–position pair as a separate dimension in a 5 × 26 = 130-dimensional vector. This orthogonal encoding lets the network cleanly distinguish and recall individual letters.

- **Clamped constraints**  
  Known letters (greens and yellows) are “anchored” by fixing their corresponding dimensions during retrieval, so the network only explores solutions that honor your Wordle feedback.

Combining Wordle’s logical pruning with Hopfield’s fast associative recall gives a powerful, brain-inspired way to suggest the best candidate words.  

## Solver Pipeline

1. **Initialization**  
   - Load your full word list (`words.txt`) and one-hot encode each 5-letter entry into a 130-dimensional vector.  
   - Instantiate the Modern Hopfield network and reset all Wordle constraints (greens, yellows, grays, and letter counts).

2. **Iterative Guess Loop**  
   1. **Enter a Guess**  
      - Type your 5-letter word.  
   2. **Record Feedback**  
      - Mark each letter as green (correct), yellow (present), or gray (absent/duplicate-excess).  
      - Update positional requirements, banned positions, and per-letter min/max counts.  
   3. **Filter Candidates**  
      - In one pass over the **entire** dictionary, discard any word that violates your accumulated constraints.  
   4. **Hopfield Recall**  
      - Reload the remaining candidates into the network.  
      - Clamp (anchor) the known letters in their fixed positions.  
      - Run the network’s retrieval dynamics to converge on stored word patterns.  
      - Score all candidates and pick the top-K matches.  
   5. **Show Suggestions**  
      - Display the highest-scoring words as your next guesses.  
   6. **Repeat**  
      - Continue until you find the solution or choose to reset the solver.

## Experiments

1. **Dense encoding (no one-hot)**  
   - Used continuous embeddings for letter-positions.  
   - **Result:** Network converged to mixed “ghost” patterns, not valid words—recall failed.

2. **Hopfield recall without constraints**  
   - Stored all words and ran retrieval on partially masked inputs.  
   - **Result:** Occasionally hit the target when the input was very close, but generally too unconstrained to be reliable.

3. **Simple include/exclude filtering**  
   - Pruned the dictionary by global “must-have” and “must-not-have” letters only.  
   - **Result:** Reduced the search space but ignored positional feedback; top-K suggestions still often incorrect.

4. **Full Wordle constraint propagation**  
   - Incorporated greens (exact positions), yellows (letter bans per position), and grays (global min/max counts with duplicate logic).  
   - **Result:** Consistently filters to a tight candidate set and the Hopfield network reliably surfaces the correct or very close words in its top-K.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/p1neapplechoco/hopfield-wordle.git
   cd hopfield-wordle
   ```

2. **Install core dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configuration**
- Try messing with the parameters inside of the `main.py` file.

4. **Run the solver**
   ```bash
   python main.py
   ```

## References

- [Hopfield Network is All You Need](https://arxiv.org/abs/2008.02217)
- [Wordle](https://www.nytimes.com/games/wordle/index.html)
- [My Hopfiled Re-Implementation](https://github.com/p1neapplechoco/hopfield)
- [Inspiration](https://github.com/sabertoaster/HopfieldWordle)
