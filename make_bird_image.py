"""
make_bird_image.py
===================

Generates a simple placeholder sprite (`bird.png`) for the env to use
when rendering.  The env will fall back to a yellow rectangle if no
file is present, so this script is purely cosmetic — drop in any 40x30
PNG you'd prefer instead.

Usage:
    python make_bird_image.py             # writes ./bird.png
    python make_bird_image.py mybird.png  # writes ./mybird.png
"""
import sys
import pygame


def make_placeholder_bird(path: str = "bird.png",
                          size: tuple = (40, 30)) -> None:
    pygame.init()
    surf = pygame.Surface(size, pygame.SRCALPHA)
    w, h = size

    # body ellipse with a darker outline
    body_rect = (0, 0, w, h)
    pygame.draw.ellipse(surf, (255, 200, 50), body_rect)
    pygame.draw.ellipse(surf, (180, 120, 30), body_rect, 2)

    # wing — a smaller ellipse offset toward the back
    wing_rect = (int(w * 0.18), int(h * 0.45), int(w * 0.45), int(h * 0.42))
    pygame.draw.ellipse(surf, (235, 170, 30), wing_rect)
    pygame.draw.ellipse(surf, (180, 120, 30), wing_rect, 1)

    # eye — white sclera + black pupil, biased toward the front
    eye_x = int(w * 0.74)
    eye_y = int(h * 0.32)
    pygame.draw.circle(surf, (255, 255, 255), (eye_x, eye_y), 4)
    pygame.draw.circle(surf, (0, 0, 0), (eye_x + 1, eye_y), 2)

    # beak — small orange triangle pointing right
    beak = [
        (w - 2,           int(h * 0.50)),
        (w + 6,           int(h * 0.46)),
        (w - 2,           int(h * 0.66)),
    ]
    pygame.draw.polygon(surf, (240, 130, 30), beak)

    pygame.image.save(surf, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "bird.png"
    make_placeholder_bird(out)
