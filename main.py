import sys
import pygame
import numpy as np
from WordleSolver import WordleSolver

WORD_FILE = "words.txt"
SCREEN_W, SCREEN_H = 800, 600
LEFT_PANEL_W = 250
TOP_K = 10
FPS = 30
CELL_SIZE = 60
WORD_LEN = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (170, 170, 170)
YELLOW = (255, 235, 0)
GREEN = (0, 200, 0)
BUTTON_BG = (200, 50, 50)
BUTTON_TEXT = WHITE

pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Wordle Helper")
clock = pygame.time.Clock()

font_small = pygame.font.SysFont(None, 24)
font_large = pygame.font.SysFont(None, 48)
font_button = pygame.font.SysFont(None, 30)


def load_words(path):
    with open(path, "r") as f:
        return [w.strip().lower() for w in f if w.strip()]


def draw_text(text, font, color, pos):
    surf = font.render(text, True, color)
    screen.blit(surf, pos)


def build_pattern(guess, feedback):
    return "".join(guess[i] if feedback[i] == 2 else "?" for i in range(WORD_LEN))


def main():
    words = load_words(WORD_FILE)
    solver = WordleSolver(words)
    suggestions = []
    scores = []

    guess = [""] * WORD_LEN
    feedback = [0] * WORD_LEN
    pos = 0
    mode = "input"

    reset_button = pygame.Rect(LEFT_PANEL_W + 10, 50, 100, 30)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

            elif event.type == pygame.KEYDOWN:
                if mode == "input":
                    if event.key == pygame.K_LEFT:
                        pos = max(0, pos - 1)
                    elif event.key == pygame.K_RIGHT:
                        pos = min(WORD_LEN - 1, pos + 1)
                    elif event.key == pygame.K_BACKSPACE:
                        if pos > 0:
                            pos -= 1
                        guess[pos] = ""
                    elif event.unicode.isalpha() and len(event.unicode) == 1:
                        guess[pos] = event.unicode.lower()
                        pos = min(WORD_LEN - 1, pos + 1)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if all(guess):
                            mode = "feedback"
                            pos = 0
                else:
                    if event.key == pygame.K_LEFT:
                        pos = max(0, pos - 1)
                    elif event.key == pygame.K_RIGHT:
                        pos = min(WORD_LEN - 1, pos + 1)
                    elif event.key in (pygame.K_SPACE, pygame.K_c):
                        feedback[pos] = (feedback[pos] + 1) % 3
                    elif event.key in (pygame.K_0, pygame.K_KP0):
                        feedback[pos] = 0
                    elif event.key in (pygame.K_1, pygame.K_KP1):
                        feedback[pos] = 1
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        feedback[pos] = 2
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        solver.update_constraint("".join(guess), np.array(feedback))
                        suggestions, scores = solver.possible_answers(
                            build_pattern(guess, feedback), top_k=TOP_K
                        )
                        guess = [""] * WORD_LEN
                        feedback = [0] * WORD_LEN
                        pos = 0
                        mode = "input"

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if reset_button.collidepoint(event.pos):
                    solver.reset()
                    suggestions = []
                    scores = []
                    guess = [""] * WORD_LEN
                    feedback = [0] * WORD_LEN
                    pos = 0
                    mode = "input"

        screen.fill(WHITE)
        pygame.draw.rect(screen, GRAY, (0, 0, LEFT_PANEL_W, SCREEN_H))
        draw_text("Candidates:", font_large, BLACK, (10, 10))

        pygame.draw.rect(screen, BUTTON_BG, reset_button)
        draw_text(
            "Reset", font_button, BUTTON_TEXT, (reset_button.x + 20, reset_button.y + 5)
        )

        for i, word in enumerate(suggestions[:TOP_K]):
            draw_text(f"{i + 1}. {word}", font_small, BLACK, (10, 90 + i * 30))

        grid_x = LEFT_PANEL_W + 50
        grid_y = 100
        for i in range(WORD_LEN):
            rect = pygame.Rect(
                grid_x + i * (CELL_SIZE + 10), grid_y, CELL_SIZE, CELL_SIZE
            )
            if mode == "feedback":
                color = (
                    GRAY if feedback[i] == 0 else YELLOW if feedback[i] == 1 else GREEN
                )
            else:
                color = WHITE
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)
            if guess[i]:
                letter = font_large.render(guess[i].upper(), True, BLACK)
                screen.blit(letter, letter.get_rect(center=rect.center))
            if i == pos:
                pygame.draw.rect(screen, BLACK, rect, 4)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
