"""
flappy_env.py
==============

Gym-style Flappy Bird environment.

The bird sits at a fixed x and accelerates downward each frame (gravity).
A "jump" action snaps its velocity to `jump_strength` (negative = up).
Pipes spawn off the right edge at a level-defined cadence, scroll left
at a constant speed, and disappear off the left.  Each pipe has a top
edge and a bottom edge — the gap between them is what the bird has to
fly through.

Episode terminates when:
  • the bird's bounding box overlaps a pipe         → done_type='hit_pipe'
  • the bird leaves the vertical screen bounds       → done_type='offscreen'
  • the bird has scored `game_length` pipes          → done_type='well_done'

The agent interface mirrors a typical gym loop but is push-based:
  env.play(player) runs one episode and calls
    player.choose_action(state, action_table)                  → action
    player.receive_after_action_observation(next_state, action_table)
"""
import copy
import math
import os
import random

import gymnasium as gym
import pygame
import yaml

from flappy_clock import ClockWrapper


class FlappyBirdEnv(gym.Env):
    def __init__(self,
                 config_file_path: str,
                 show_screen: bool = None,
                 level: int = None,
                 game_length: int = None,
                 verbose: bool = None,
                 random_seed: int = None):
        # ------------------------------------------------------------------
        # Load config.
        # ------------------------------------------------------------------
        if not config_file_path or not os.path.exists(config_file_path):
            raise FileNotFoundError(f"config file not found: {config_file_path!r}")
        with open(config_file_path, "r") as fh:
            cfg = yaml.safe_load(fh)

        # ------------------------------------------------------------------
        # Pull globals out of the config.
        # ------------------------------------------------------------------
        self.screen_width = cfg['screen_width']
        self.screen_height = cfg['screen_height']
        self.frame_rate = cfg['frame_rate']
        self.gravity = cfg['gravity']
        self.jump_strength = cfg['jump_strength']
        self.pipe_speed = cfg['pipe_speed']
        self.colors = cfg['colors']
        self.action_table = cfg['action_table']
        self.bird_attributes = cfg['bird_attributes']
        self.bird_img_path = cfg.get('bird_img_path', 'bird.png')

        # ------------------------------------------------------------------
        # Level selection — every kwarg passed to __init__ takes priority
        # over the corresponding YAML default.
        # ------------------------------------------------------------------
        self.level = level if level is not None else cfg['level']
        lvl_cfg = cfg['levels'][self.level]
        self.pipe_attributes = lvl_cfg['pipe_attributes']
        self.pipe_frequency = self.pipe_attributes['pipe_frequency']
        self.minimum_action_gap = lvl_cfg['minimum_action_gap']
        self.game_length = (game_length
                            if game_length is not None
                            else lvl_cfg.get('game_length', 10))

        self.show_screen = (show_screen
                            if show_screen is not None
                            else cfg.get('show_screen', False))
        self.verbose = (verbose
                        if verbose is not None
                        else cfg.get('verbose', False))

        # Seeded RNG so headless training is reproducible.
        seed = random_seed if random_seed is not None else cfg.get('random_seed', 0)
        self.rng = random.Random(seed)

        # ------------------------------------------------------------------
        # Reverse lookup for the action table — used for verbose rendering.
        # ------------------------------------------------------------------
        self._action_name = {v: k for k, v in self.action_table.items()}

        # ------------------------------------------------------------------
        # Mutable game state.  reset() will reinitialise everything below.
        # ------------------------------------------------------------------
        self.bird_x = self.bird_attributes['x']
        self.bird_y = self.bird_attributes['y']
        self.bird_velocity = 0.0
        self.pipes: list = []
        self.score = 0
        self.mileage = 0
        self.done = False
        self.done_type = None
        self._last_spawn_time = 0
        self._episode_start_time = 0

        # rendering handles — created lazily inside play() when show_screen
        # is True, since headless training shouldn't touch pygame display.
        self.screen = None
        self.font = None
        self.bird_font = None
        self.bird_img = None
        self._bird_img_obj = None     # raw, unscaled sprite

        self.clock = ClockWrapper(show_screen=self.show_screen,
                                  frame_rate=self.frame_rate)

    # ----------------------------------------------------------------------
    # public API used by the agent / training loop
    # ----------------------------------------------------------------------
    def reset_random_seed(self, random_seed: int = None) -> None:
        """Re-seed the pipe RNG.  Handy for evaluation, where you want
        each eval episode to roll its own pipe sequence so the agent
        can't overfit to a single deterministic layout."""
        self.rng = random.Random(random_seed)

    def reset(self, **kwargs):
        # Gymnasium asks subclasses to call super().reset(**kwargs) here.
        super().reset(**kwargs)
        self.bird_x = self.bird_attributes['x']
        self.bird_y = self.bird_attributes['y']
        self.bird_velocity = 0.0
        self.pipes = []
        self.score = 0
        self.mileage = 0
        self.done = False
        self.done_type = None
        self._last_spawn_time = 0
        self._episode_start_time = 0
        # rebuild the clock so headless time starts at zero for every episode
        self.clock = ClockWrapper(show_screen=self.show_screen,
                                  frame_rate=self.frame_rate)
        return self.get_state()

    def get_state(self) -> dict:
        """Return a snapshot of everything an agent might want.

        Deep-copied so an agent can't mutate the live env state by
        holding references.  The schema is what the DQN script expects:
        keys like bird_y, bird_velocity, pipes, score, etc.
        """
        return copy.deepcopy({
            'bird_x':         self.bird_x,
            'bird_y':         self.bird_y,
            'bird_width':     self.bird_attributes['width'],
            'bird_height':    self.bird_attributes['height'],
            'bird_velocity':  self.bird_velocity,
            'pipes':          self.pipes,
            'pipe_attributes': self.pipe_attributes,
            'screen_width':   self.screen_width,
            'screen_height':  self.screen_height,
            'score':          self.score,
            'mileage':        self.mileage,
            'done':           self.done,
            'done_type':      self.done_type,
        })

    def get_action_table(self) -> dict:
        return copy.deepcopy(self.action_table)

    # ----------------------------------------------------------------------
    # main loop
    # ----------------------------------------------------------------------
    def play(self, player=None) -> None:
        """Run a single episode driven by `player`.

        The player is expected to implement:
            choose_action(state, action_table) -> action_code
            receive_after_action_observation(next_state, action_table) -> None
        """
        # Player can flip the env's render mode for this episode (the
        # DQN agent does this when entering eval).
        if hasattr(player, 'show_screen'):
            self.show_screen = bool(player.show_screen)

        self.reset()

        if self.show_screen:
            self._init_render_surface()

        self._last_spawn_time = self.clock.current_time()
        self._episode_start_time = self._last_spawn_time

        while not self.done:
            if self.show_screen:
                self.screen.fill(self.colors['background'])

            current = self.get_state()
            action = player.choose_action(
                state=current,
                action_table=self.get_action_table(),
            )

            # Translate action code into bird state.
            if action == self.action_table['quit_game']:
                self.done = True
            elif action == self.action_table['jump']:
                self.bird_velocity = self.jump_strength
            elif action == self.action_table['do_nothing']:
                pass
            else:
                raise ValueError(f"Invalid action code: {action!r}")

            # Run `minimum_action_gap` physics steps before the agent
            # gets to act again.  The first step uses the agent's chosen
            # action; the rest are forced to "do_nothing".
            for i in range(self.minimum_action_gap):
                self.step(action if i == 0 else self.action_table['do_nothing'])

            # Stop when the agent has cleared enough pipes.
            if self.score >= self.game_length:
                self.done = True
                self.done_type = 'well_done'

            after = self.get_state()
            player.receive_after_action_observation(
                after,
                action_table=self.get_action_table(),
            )

    # ----------------------------------------------------------------------
    # single physics step
    # ----------------------------------------------------------------------
    def step(self, action) -> None:
        """Advance one physics frame.

        Side effects in order:
          1. gravity applied to bird velocity, velocity applied to y
          2. spawn a new pipe if pipe_frequency ms have elapsed since
             the last spawn
          3. scroll all pipes left by pipe_speed; tally mileage
          4. remove pipes that have gone off-screen left (= +1 score each)
          5. collision detection against any pipe currently overlapping
             the bird in x
          6. offscreen check (top OR bottom of screen)
          7. if rendering, draw the frame
          8. tick the clock
        """
        if self.show_screen:
            self.screen.fill(self.colors['background'])

        now = self.clock.current_time()

        # ---- 1. bird physics ----------------------------------------------
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity

        # ---- 2. pipe spawn ------------------------------------------------
        if now - self._last_spawn_time > self.pipe_frequency:
            self._spawn_pipe(now)
            self._last_spawn_time = now

        # ---- 3. scroll pipes ----------------------------------------------
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_speed
        self.mileage += self.pipe_speed

        # ---- 4. cull off-screen pipes; each one is a +1 score -------------
        pipe_w = self.pipe_attributes['width']
        kept = [p for p in self.pipes if p['x'] + pipe_w > 0]
        self.score += len(self.pipes) - len(kept)
        self.pipes = kept

        # ---- 5. collision -------------------------------------------------
        bird_w = self.bird_attributes['width']
        bird_h = self.bird_attributes['height']
        bird_right = self.bird_x + bird_w
        bird_bot = self.bird_y + bird_h
        for pipe in self.pipes:
            in_x = bird_right > pipe['x'] and self.bird_x < pipe['x'] + pipe_w
            if in_x:
                if self.bird_y < pipe['top'] or bird_bot > pipe['bottom']:
                    self.done_type = 'hit_pipe'
                    self.done = True
                    break

        # ---- 6. offscreen check ------------------------------------------
        if self.bird_y > self.screen_height or self.bird_y < 0:
            self.done_type = 'offscreen'
            self.done = True

        # If neither of the termination conditions fired, mark the step as
        # an ordinary "still alive" frame so done_type stays a useful
        # discriminator for the reward function downstream.
        if not self.done:
            self.done_type = 'not_done'

        # ---- 7. render ----------------------------------------------------
        if self.show_screen:
            self._draw_frame(action)

        # ---- 8. clock -----------------------------------------------------
        self.clock.tick()

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def _spawn_pipe(self, now: int) -> None:
        """Append a new pipe to the right edge with a vertically-randomised
        gap.  Two formation modes are supported, set per-level in the YAML:

          random : top is uniform on [mean - offset, mean + offset]
          sine   : top oscillates smoothly with time
        """
        attrs = self.pipe_attributes
        formation = attrs['formation']
        mean = attrs['window_y_mean']
        offset = attrs['window_y_offset']
        gap = attrs['gap']
        width = attrs['width']

        if formation == 'random':
            if offset == 0:
                top = mean
            else:
                top = self.rng.randint(mean - offset, mean + offset)
        elif formation == 'sine':
            elapsed = now - self._episode_start_time
            phase = (elapsed / max(self.pipe_frequency, 1) - 1.0) * (math.pi / 4.0)
            top = math.sin(phase) * offset + mean
        else:
            raise ValueError(f"Unknown pipe formation: {formation!r}")

        self.pipes.append({
            "x":      self.screen_width,
            "top":    top,
            "bottom": top + gap,
            "width":  width,
        })

    def _init_render_surface(self) -> None:
        """Set up pygame display + sprite the first time we render."""
        pygame.init()
        if self.screen is None:
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Flappy Bird DQN")
        if self.font is None:
            self.font = pygame.font.Font(None, 30)
            self.bird_font = pygame.font.Font(None, 24)
        # try to load the bird sprite once; fall back to a coloured rect.
        if self._bird_img_obj is None and os.path.exists(self.bird_img_path):
            try:
                self._bird_img_obj = pygame.image.load(self.bird_img_path)
            except Exception:
                self._bird_img_obj = None
        if self._bird_img_obj is not None and self.bird_img is None:
            self.bird_img = pygame.transform.scale(
                self._bird_img_obj,
                (self.bird_attributes['width'], self.bird_attributes['height']),
            )

    def _draw_frame(self, action) -> None:
        """Draw pipes + bird + scoreboard for the current state."""
        pipe_w = self.pipe_attributes['width']
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, self.colors['pipe'],
                             (pipe['x'], 0, pipe_w, pipe['top']))
            pygame.draw.rect(self.screen, self.colors['pipe'],
                             (pipe['x'], pipe['bottom'], pipe_w,
                              self.screen_height - pipe['bottom']))

        if self.bird_img is not None:
            self.screen.blit(self.bird_img, (self.bird_x, self.bird_y))
        else:
            # Fallback: a yellow rectangle.  Looks like Pac-Man, gets
            # the point across.
            pygame.draw.rect(
                self.screen, (255, 200, 50),
                (self.bird_x, self.bird_y,
                 self.bird_attributes['width'], self.bird_attributes['height']),
            )

        scoreboard = self.font.render(
            f"Mileage: {self.mileage}   Score: {self.score}",
            True, self.colors['score'],
        )
        self.screen.blit(scoreboard, (10, 10))

        if self.verbose:
            info = self.bird_font.render(
                f"(y={int(self.bird_y):03d}, v={int(self.bird_velocity):02d}, "
                f"a={self._action_name.get(action, '?')})",
                True, self.colors['score'],
            )
            self.screen.blit(info, (self.bird_x, max(0, self.bird_y - 24)))

        pygame.display.update()
        # pump the event queue so the OS doesn't decide the window is
        # unresponsive during long runs
        pygame.event.pump()
