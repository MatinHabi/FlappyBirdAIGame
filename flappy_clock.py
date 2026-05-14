"""
flappy_clock.py
================

Tiny clock wrapper for the Flappy Bird environment.

The environment runs in two modes:
  * Rendered      — needs real-time pacing so animation is watchable.
  * Headless RL   — runs as fast as the CPU allows; "time" only matters
                    insofar as the pipe-spawn cadence is measured in ms.

Rather than scattering branches everywhere, this class hides the
distinction behind a `current_time()` / `tick()` interface:

  • If `show_screen` is True, current_time falls through to pygame's
    real-time millisecond ticker, and tick() caps the loop at frame_rate.
  • If `show_screen` is False, current_time reads from a deterministic
    counter that advances by exactly 1000/frame_rate ms every tick().
    No display, no real-time waiting — but training stays reproducible.
"""
import pygame


class ClockWrapper:
    def __init__(self, show_screen: bool = False, frame_rate: int = 30):
        self.show_screen = bool(show_screen)
        self.frame_rate = int(frame_rate)
        # the ms-per-frame increment we use for the headless counter
        self._ms_per_frame = int(round(1000.0 / max(self.frame_rate, 1)))
        # synthetic millisecond clock used when running headless
        self._headless_ms = 0
        # real pygame clock used only when rendering
        self._py_clock = pygame.time.Clock() if self.show_screen else None

    def current_time(self) -> int:
        """Return the current "wall clock" time in milliseconds.

        Headless: a deterministic integer counter advanced by tick().
        Rendered: pygame's real-time ticker.
        """
        if self.show_screen:
            return pygame.time.get_ticks()
        return self._headless_ms

    def tick(self) -> None:
        """Advance one frame.  In rendered mode this also sleeps the
        thread enough to hold the frame_rate cap; headless mode just
        bumps the synthetic counter."""
        if self.show_screen:
            self._py_clock.tick(self.frame_rate)
        else:
            self._headless_ms += self._ms_per_frame
