from WordleSolver import WordleWSolver

with open("words.txt", "r") as f:
    word_pool = [w.strip() for w in f]

ws = WordleWSolver(words=word_pool, beta=10000)

word_to_solve = "ba?dy"

include = list("a")
exclude = list("snesmrhpl")

ws.add_constraint(include=include, exclude=exclude)

words, scores = ws.possible_answers(word_to_solve=word_to_solve, top_k=5)

for rank, (word, s) in enumerate(zip(words, scores), 1):
    print(f" {rank}. {word}  (score={s:.3f})")
