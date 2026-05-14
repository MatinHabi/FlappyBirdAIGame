"""
sec_model.py
=============

Double-DQN agent for the Flappy Bird DQN environment.

Highlights
----------
  • 6-dim state representation with `gap_size` so the network can
    distinguish per-level difficulty (L1's 600px gap vs L5's 150px).
  • Two-buffer replay: a large FIFO buffer (`storage`) for all
    transitions, plus a small "event buffer" that only holds deaths
    and pipe-pass moments.  Each minibatch is 75% main + 25% event so
    the agent never forgets how to die — even after thousands of safe
    flying frames would otherwise flush those frames out.
  • Gap-aware Gaussian centring reward — the bonus's bell width scales
    with the gap, so the same shaping function works for L1 and L5.
  • Double DQN (action selection with the main net, value with the
    target net) plus on-the-fly q_next computation against the CURRENT
    target net — no stale cached bootstrap targets.
  • Periodic honest-eval (pure-greedy, 30 eps per level) gates writes
    to `sec_model.ckpt`, separate from the running training-best save
    in `sec_model_running.ckpt`.
  • Optional live matplotlib dashboard + CSV streams when --csv is on.
"""
import numpy as np
import pygame
from mlp_regressor import MLPRegression
import argparse
from flappy_env import FlappyBirdEnv
from collections import deque
from colorama import Fore, Style, init
import random
import csv  # for --csv real-time training-metric logging

# ##############################################################
# ##  Optional matplotlib for the LIVE TRAINING DASHBOARD that ##
# ##  pops open when --csv is passed.  If matplotlib isn't     ##
# ##  available we fall back to CSV-only — no crash.           ##
# ##############################################################
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # modify these
        # ##############################################################
        # ##  TWO-STORAGE REPLAY                                        ##
        # ##  ----------------------                                    ##
        # ##  storage:      main FIFO buffer of all transitions.        ##
        # ##  event_buffer: HIGH-IMPACT transitions only — deaths and   ##
        # ##               pipe-pass moments.  Once a "best/worst"      ##
        # ##               state enters this buffer it isn't flushed by ##
        # ##               thousands of boring safe-flight frames.      ##
        # ##  Every minibatch is 75% main + 25% event_buffer so the     ##
        # ##  network always sees fresh "what crashed me" gradients     ##
        # ##  even when the agent is mostly succeeding.  This is the    ##
        # ##  single most effective counter to catastrophic forgetting  ##
        # ##  in this game.                                             ##
        # ##############################################################
        self.storage = deque(maxlen=100000)        # main replay (FIFO over all transitions)
        self.event_buffer = deque(maxlen=5000)     # deaths + pipe-passes only
        self.event_batch_ratio = 0.25              # 25% of each minibatch comes from event_buffer

        # A neural network MLP model which can be used as Q
        # ##############################################################
        # ##  INPUT DIM = 6.  Slot map:                                 ##
        # ##    [0] bird_y         (0..1)                               ##
        # ##    [1] bird_vel       (clipped /10)                        ##
        # ##    [2] dist_to_pipe   (pipe-1 x − bird_x) / screen_w       ##
        # ##    [3] relative1      bird_cy − gap-centre-of-pipe-1       ##
        # ##    [4] relative2      bird_cy − gap-centre-of-pipe-2       ##
        # ##    [5] gap_size       pipe-1 gap height                    ##
        # ##                                                            ##
        # ##  gap_size is the killer feature — it lets the network     ##
        # ##  pick L1's "any altitude is fine" policy apart from L5's   ##
        # ##  "thread the needle" policy without compromising.          ##
        # ##############################################################
        self.network = MLPRegression(input_dim=6, output_dim=2, learning_rate=0.0003)
        self.network2 = MLPRegression(input_dim=6, output_dim=2, learning_rate=0.0003)
        # initialise Q_f's parameter by Q's, here is an example
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        #Hyperparameters
        self.epsilon = 1.0  # probability ε in Algorithm 2
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.02
        self.n = 64
        self.discount_factor = 0.99  # γ in Algorithm 2
        self._prev_score = 0

        #variables to store in transitions every frame
        self.phi_t = None
        self.action_idx = None
        self.episode_count = 0

        # ##############################################################
        # ##  WARMUP — don't take any gradient steps until the buffer  ##
        # ##  has WARMUP_FRAMES varied transitions.  Otherwise the     ##
        # ##  first few thousand updates fit to whatever happens to be ##
        # ##  in the buffer (often: dying-on-spawn) and that bias is   ##
        # ##  hard to undo later.                                       ##
        # ##############################################################
        self.warmup_frames = 5000

        self.train_every_k_frames = 4
        self._frame_counter = 0
        self.y_clip_min = -20.0
        self.y_clip_max = 100.0

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)

    def build_state(self, state: dict):
        """6-dim state representation.

        Slot map (must match input_dim=6 in __init__):
          [0] bird_y         : absolute screen position, 0..1
          [1] bird_vel       : velocity / 10, clipped to [-1, 1]
          [2] dist_to_pipe   : (next pipe x - bird_x) / screen_w
          [3] relative1      : (bird_centre_y - gap_centre_of_pipe1) / screen_h
          [4] relative2      : (bird_centre_y - gap_centre_of_pipe2) / screen_h
                                (falls back to relative1 if pipe 2 isn't visible)
          [5] gap_size       : (pipe_bottom - pipe_top) / screen_h
        """
        screen_h = state['screen_height']
        screen_w = state['screen_width']

        bird_y = state['bird_y'] / screen_h
        # /10 gives bird_vel twice the dynamic range over the bird's
        # actual operating envelope vs the /20 we tried earlier.
        bird_vel = max(-1.0, min(1.0, state['bird_velocity'] / 10.0))

        pipes = state['pipes']
        bird_x = state['bird_x']
        bird_centre_y = state['bird_y'] + state['bird_height'] / 2.0

        # Defensive sort — the env already returns pipes left-to-right
        # but the pipe-scan loop below depends on that ordering so we
        # belt-and-brace it.
        pipes_sorted = sorted(pipes, key=lambda p: p['x'])

        # Find the first two pipes still ahead of the bird.
        next_pipe = None
        next2_pipe = None
        for pipe in pipes_sorted:
            if pipe['x'] + pipe['width'] > bird_x:
                if next_pipe is None:
                    next_pipe = pipe
                else:
                    next2_pipe = pipe
                    break

        if next_pipe is None:
            # No pipes yet — neutral "nothing to worry about" defaults.
            # gap_size default ≈ 0.25 (the tightest level's gap),
            # which biases the agent toward conservative flying when blind.
            dist_to_pipe = 1.0
            relative1    = 0.0
            relative2    = 0.0
            gap_size     = 0.25
        else:
            dist_to_pipe = (next_pipe['x'] - bird_x) / screen_w
            gap_centre1  = (next_pipe['top'] + next_pipe['bottom']) / 2.0
            relative1    = (bird_centre_y - gap_centre1) / screen_h
            gap_size     = (next_pipe['bottom'] - next_pipe['top']) / screen_h

            if next2_pipe is None:
                # only one pipe visible — pretend pipe 2 is in the same place
                relative2 = relative1
            else:
                gap_centre2 = (next2_pipe['top'] + next2_pipe['bottom']) / 2.0
                relative2   = (bird_centre_y - gap_centre2) / screen_h

        phi = np.array([bird_y, bird_vel, dist_to_pipe,
                        relative1, relative2, gap_size])
        return phi.reshape(1, -1)

    def reward(self, state: dict, phi: np.ndarray):
        """Reward function with GAUSSIAN centring shaping.

          (a) The Gaussian e^(-k·x²) bonus has a sharp peak at relative1=0
              and falls off smoothly, so the gradient is strongest where the
              agent is *almost* perfectly centred — exactly where it needs
              to learn fine motor control.  Linear-decay flattens out, so
              the agent gets "good enough" reward across a wider band and
              never tightens up.
          (b) The shaping uses gap_size to ADAPT the tolerance:
              wider gaps tolerate larger relative1 values without penalty,
              tight gaps demand precision.  This unlocks per-level policy.
        """
        done_type = state['done_type']

        # Sharper death penalties — dying should be MUCH worse than
        # collecting a few crumbs of alive-reward, so "stay alive" wins
        # the Q argmax.
        if done_type == 'offscreen':
            return -10.0
        if done_type == 'hit_pipe':
            return -5.0

        r = 0.04                                        # small alive crumb

        current_score = state['score']
        if current_score > self._prev_score:
            r += 5.0                                    # pipe-pass spike
        self._prev_score = current_score

        # ##############################################################
        # ##  GAP-AWARE GAUSSIAN CENTRING                              ##
        # ##  The 1/(gap_size+eps) scale shrinks the Gaussian's        ##
        # ##  width when the gap is tight (e.g. L5: gap_size=0.25) and ##
        # ##  widens it when the gap is huge (L1: gap_size=1.0).  So   ##
        # ##  the agent gets paid for "centred within the gap" rather  ##
        # ##  than "centred absolutely", and that's a fairer signal.   ##
        # ##############################################################
        dist_to_pipe = float(phi[0, 2])
        if dist_to_pipe < 1.0:                          # an actual pipe is visible
            relative1 = float(phi[0, 3])                # bird vs pipe-1 gap centre
            gap_size  = float(phi[0, 5])
            # scale: when gap is small, k is big => narrow Gaussian => need precise centring
            #        when gap is large, k is small => wide Gaussian => sloppy ok
            k1 = 1.5 / max(gap_size, 0.05) ** 2
            r += 1.0 * np.exp(-k1 * relative1 * relative1)

            # Look-ahead bonus for being aligned with pipe 2 as we exit pipe 1.
            # Same gap-aware scaling philosophy.
            if dist_to_pipe < 0.25:
                relative2 = float(phi[0, 4])
                r += 0.5 * np.exp(-k1 * relative2 * relative2)

        return r


    #choose_action (epsilon-greedy policy)
    def choose_action(self, state: dict, action_table: dict) -> int:
        """
        This function should be called when the agent action is requested.
        Args:
            state: input state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            action: the action code as specified by the action_table
        """
        phi = self.build_state(state)
        self.phi_t = phi
        if state['mileage'] == 0 and state['score'] == 0:
            self._prev_score = 0

        if self.mode == 'eval':
            #no training when the mode is "eval"
            q_vals = self.network.predict(phi)
            a_t = int(np.argmax(q_vals))
        else:
            #epsilon-greedy
            if random.random() < self.epsilon:
                # asymmetric exploration: jump much less often than do_nothing,
                # because gravity is already pulling the bird down
                a_t = np.random.choice([0,1], p=[0.05,0.95])
            else:
                q_vals = self.network.predict(phi)
                a_t = int(np.argmax(q_vals))

        self.action_idx = a_t

        return action_table['jump'] if a_t == 0 else action_table['do_nothing']

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        """
        Post-action callback.  Stores the transition, applies the K-frame
        training throttle, and runs one Double-DQN gradient step.
        """
        if self.mode != 'train':
            return

        s_t_next = self.build_state(state)
        r_t = self.reward(state, s_t_next)
        terminal = bool(state['done'])
        done_type = state.get('done_type')

        transition = (self.phi_t.copy(), self.action_idx, r_t, s_t_next.copy(), terminal)

        # Main buffer — every transition goes here.
        self.storage.append(transition)

        # ##############################################################
        # ##  EVENT BUFFER — only HIGH-IMPACT transitions.             ##
        # ##  We capture:                                              ##
        # ##    • Deaths (terminal due to offscreen/hit_pipe).         ##
        # ##    • Pipe-pass frames (reward > 4 means the +5 bonus      ##
        # ##      fired this frame).                                   ##
        # ##  Both extremes carry strong learning signal.  By keeping  ##
        # ##  them in a separate 5k-deep buffer we guarantee they      ##
        # ##  survive long stretches of safe-flight FIFO churn.        ##
        # ##############################################################
        if terminal and done_type in ('offscreen', 'hit_pipe'):
            self.event_buffer.append(transition)
        elif r_t > 4.0:                                  # caught a pipe-pass (+5 spike)
            self.event_buffer.append(transition)

        # K-frame training throttle.
        self._frame_counter += 1
        if self._frame_counter % self.train_every_k_frames != 0:
            return

        # Warmup: don't train until the buffer has some diversity.  Otherwise
        # the first few thousand updates lock into whatever's randomly there.
        if self._frame_counter < self.warmup_frames:
            return

        if len(self.storage) < self.n:
            return

        # ##############################################################
        # ##  MIXED MINIBATCH — 75% main buffer + 25% event buffer.    ##
        # ##  When event_buffer is too small we fall back to a pure-   ##
        # ##  main sample (typical at the start of training).          ##
        # ##############################################################
        n_event = min(int(self.n * self.event_batch_ratio), len(self.event_buffer))
        n_main  = self.n - n_event
        minibatch = random.sample(self.storage, n_main)
        if n_event > 0:
            minibatch = minibatch + random.sample(self.event_buffer, n_event)

        # ##############################################################
        # ##  BATCHED DOUBLE-DQN TARGET COMPUTATION                    ##
        # ##  -------------------------------------                    ##
        # ##  Double DQN: pick a' with the MAIN net, evaluate Q(s',a') ##
        # ##  with the TARGET net.  Standard DQN's max-over-target     ##
        # ##  overestimates Q (Hasselt 2015) — and overestimation is   ##
        # ##  the death-spiral mode where the net keeps inflating Q on ##
        # ##  bad actions until policy collapses.                      ##
        # ##                                                            ##
        # ##  Batched into a single forward pass per network instead   ##
        # ##  of one-at-a-time so the K=4 throttle doesn't hurt        ##
        # ##  wall-clock training speed.                                ##
        # ##############################################################
        S_next = np.vstack([trans[3] for trans in minibatch])         # (n, 6)
        terminals = np.array([trans[4] for trans in minibatch], dtype=bool)

        Q_main_next = self.network.predict(S_next)                    # (n, 2)
        a_next = np.argmax(Q_main_next, axis=1)                       # (n,)
        Q_target_next = self.network2.predict(S_next)                 # (n, 2)
        q_next_vals = Q_target_next[np.arange(self.n), a_next]        # (n,)
        q_next_vals = np.where(terminals, 0.0, q_next_vals)           # zero out terminals

        X = [] #phi_j
        Y = [] #bellman targets
        W = [] #onehot vector mask

        for i, (phi_j, action_j, r_j, _, _) in enumerate(minibatch):
            # Bellman target
            y_j = r_j + self.discount_factor * float(q_next_vals[i])

            # Clip the Bellman target — a single outlier r+γQ' value can
            # otherwise drag the whole network during a fit_step.
            y_j = max(self.y_clip_min, min(self.y_clip_max, y_j))

            w_j = np.zeros(2)
            w_j[action_j] = 1.0

            X.append(phi_j.flatten())
            Y.append([y_j, y_j])
            W.append(w_j)

        X = np.array(X)
        Y = np.array(Y)
        W = np.array(W)

        self.network.fit_step(X, Y, W) #one gradient step on Q

    def save_model(self, path: str = 'sec_model.ckpt'):
        self.network.save_model(path=path)

    def load_model(self, path: str = 'sec_model.ckpt'):
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to a checkpoint to resume training from.')
    parser.add_argument('--eval', action='store_true',
                        help='Skip training and watch the saved model play.')
    parser.add_argument('--genTrain', action='store_true', help='Train on all levels')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes.')
    parser.add_argument('--model-path', type=str, default='sec_model.ckpt')
    # ##############################################################
    # ##  --csv : stream per-episode + per-eval metrics into two   ##
    # ##  CSV files (fun_train.csv, fun_eval.csv) that are flushed ##
    # ##  every row + opens a live matplotlib dashboard window.    ##
    # ##############################################################
    parser.add_argument('--csv', action='store_true',
                        help='Stream training/eval metrics + open a live dashboard.')
    args = parser.parse_args()

    # All six levels share game_length=10 in this consolidated config, so
    # episode scores are directly comparable.
    game_length = 10

    # ##############################################################
    # ##  PERIODIC HONEST-EVAL  →  sec_model.ckpt                   ##
    # ##  Every EVAL_EVERY training eps we run EVAL_EPS_PER_LEVEL   ##
    # ##  pure-greedy episodes (mode='eval') per level and only     ##
    # ##  overwrite sec_model.ckpt when the WORST-LEVEL eval avg    ##
    # ##  beats the previous best.  Pure-greedy means ε is ignored, ##
    # ##  so the score reflects the actual policy.                  ##
    # ##  sec_model_running.ckpt tracks the in-training best        ##
    # ##  (good for `--load` resume); sec_model.ckpt is the         ##
    # ##  verified gold copy.                                       ##
    # ##############################################################
    EVAL_EVERY = 1000
    EVAL_EPS_PER_LEVEL = 30

    def eval_agent_on_levels(agent, levels, eps_per_level=EVAL_EPS_PER_LEVEL):
        """Pure-greedy eval over each level → {level: mean_score}.

        All levels share game_length=10 in this config, so we don't need
        any per-level game_length branching.
        """
        saved_mode = agent.mode
        saved_show = agent.show_screen
        saved_prev_score = agent._prev_score
        agent.mode = 'eval'
        agent.show_screen = False
        out = {}
        for lvl in levels:
            env_eval = FlappyBirdEnv(config_file_path='config.yml', show_screen=False,
                                     level=lvl, game_length=10)
            scores = []
            for _ in range(eps_per_level):
                env_eval.play(player=agent)
                scores.append(env_eval.score)
            out[lvl] = float(np.mean(scores))
        agent.mode = saved_mode
        agent.show_screen = saved_show
        agent._prev_score = saved_prev_score
        return out

    # ----- CSV streaming setup --------------------------------------------
    csv_train_file = None
    csv_train_writer = None
    csv_eval_file = None
    csv_eval_writer = None
    if args.csv:
        csv_train_file = open('fun_train.csv', 'w', newline='')
        csv_train_writer = csv.writer(csv_train_file)
        csv_train_writer.writerow([
            'episode', 'level', 'score', 'mileage', 'epsilon',
            'L1_avg', 'L2_avg', 'L3_avg', 'L4_avg', 'L5_avg', 'L6_avg',
            'running_avg', 'best_worst_avg',
        ])
        csv_train_file.flush()
        csv_eval_file = open('fun_eval.csv', 'w', newline='')
        csv_eval_writer = csv.writer(csv_eval_file)
        csv_eval_writer.writerow([
            'episode',
            'L1_eval', 'L2_eval', 'L3_eval', 'L4_eval', 'L5_eval', 'L6_eval',
            'eval_worst', 'best_eval_worst', 'saved_sec_model',
        ])
        csv_eval_file.flush()

    best_eval_worst_avg = -1

    # ----- live dashboard --------------------------------------------------
    PLOT_UPDATE_EVERY = 50
    plotter = None
    if args.csv and HAS_MATPLOTLIB:
        try:
            plt.ion()
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            fig.suptitle('Flappy Bird DQN — live training dashboard', fontsize=13)
            plotter = {
                'fig': fig,
                'axes': axes,
                'last_drawn_ep': -PLOT_UPDATE_EVERY,
                'history': {
                    'episodes':    [],
                    'L1': [], 'L2': [], 'L3': [], 'L4': [], 'L5': [], 'L6': [],
                    'epsilon':     [],
                    'running_avg': [],
                    'eval_eps':    [],
                    'eval_worst':  [],
                    'eval_saved':  [],
                },
            }
            fig.canvas.draw_idle()
            try:
                fig.canvas.flush_events()
            except Exception:
                pass
            print(Fore.CYAN + "[plot] live dashboard window opened — "
                              "leave it visible during training" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.YELLOW + f"[plot] matplotlib couldn't open a window "
                                f"({e}); falling back to CSV-only." + Style.RESET_ALL)
            plotter = None
    elif args.csv and not HAS_MATPLOTLIB:
        print(Fore.YELLOW + "[plot] matplotlib not installed — "
                            "CSVs will still be written, but no live window." + Style.RESET_ALL)

    def refresh_dashboard(plotter):
        if plotter is None:
            return
        h = plotter['history']
        axes = plotter['axes']
        for ax in axes.flat:
            ax.clear()

        # (0,0) per-level rolling averages (6 lines, L1..L6)
        ax = axes[0, 0]
        for key, color in zip(('L1', 'L2', 'L3', 'L4', 'L5', 'L6'),
                              ('tab:blue', 'tab:orange', 'tab:green',
                               'tab:red', 'tab:purple', 'tab:brown')):
            ax.plot(h['episodes'], h[key], label=key, color=color, linewidth=1.2)
        ax.set_title('Per-level rolling avg (last 100 eps)')
        ax.set_xlabel('episode'); ax.set_ylabel('avg score')
        ax.set_ylim(bottom=0)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

        # (0,1) epsilon
        ax = axes[0, 1]
        ax.plot(h['episodes'], h['epsilon'], color='tab:purple')
        ax.set_title('ε (exploration probability)')
        ax.set_xlabel('episode'); ax.set_ylabel('ε')
        ax.grid(alpha=0.3)

        # (1,0) overall 500-ep running average
        ax = axes[1, 0]
        ax.plot(h['episodes'], h['running_avg'], color='tab:orange')
        ax.set_title('Running average (last 500 eps, all levels)')
        ax.set_xlabel('episode'); ax.set_ylabel('avg score')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

        # (1,1) eval worst + sec_model.ckpt save markers
        ax = axes[1, 1]
        if h['eval_eps']:
            ax.plot(h['eval_eps'], h['eval_worst'], color='tab:red',
                    marker='o', markersize=4, linewidth=1.2,
                    label='worst-level eval')
            save_eps    = [e for e, s in zip(h['eval_eps'], h['eval_saved']) if s]
            save_worsts = [w for w, s in zip(h['eval_worst'], h['eval_saved']) if s]
            if save_eps:
                ax.scatter(save_eps, save_worsts, color='magenta', s=80,
                           marker='*', label='SAVED sec_model.ckpt', zorder=5)
            ax.legend(loc='lower right', fontsize=8)
        ax.set_title('Eval: worst-level mean per checkpoint')
        ax.set_xlabel('episode'); ax.set_ylabel('worst eval avg')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

        plotter['fig'].tight_layout(rect=[0, 0, 1, 0.96])
        try:
            plotter['fig'].canvas.draw_idle()
            plotter['fig'].canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

    # ----- --eval branch ---------------------------------------------------
    if args.eval:
        env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=True,
                             level=args.level, game_length=game_length)
        agent2 = MyAgent(show_screen=True, load_model_path=args.model_path, mode='eval')

        scores = []
        for episode in range(args.episodes):
            env2.play(player=agent2)
            scores.append(env2.score)
            print(f"Episode {episode:3d} | Score: {env2.score} | Mileage: {env2.mileage}")

        print(f"\nMax score:  {np.max(scores)}")
        print(f"Mean score: {np.mean(scores):.2f}")
        raise SystemExit

    # ----- training scaffolding -------------------------------------------
    episodes = 70000 if args.genTrain else 50000
    best_score = -1
    best_running_average = -1
    running_average = 0
    recent = deque(maxlen=500)
    ra_history = deque(maxlen=100)

    per_level_recent = {lvl: deque(maxlen=100) for lvl in range(1, 7)}
    per_level_avg_history = {l: deque(maxlen=20) for l in range(1, 7)}
    best_worst_avg = -1 if not args.load else 2.02

    # ##############################################################
    # ##                        genTrain                          ##
    # ##  Trains over a weighted mix of L1..L6, biased toward     ##
    # ##  whichever level is currently doing worst.               ##
    # ##############################################################
    if args.genTrain:

        agent = MyAgent(show_screen=args.watch, load_model_path=args.load)

        if args.load:
            MyAgent.update_network_model(net_to_update=agent.network2, net_as_source=agent.network)
            agent.epsilon = 0.1

        def color_for(score, lvl):
            # All levels share game_length=10, so the same thresholds work.
            # 9..10 is near-perfect, 7-8 is great, 4-6 is decent, below is poor.
            if   score >= 9:  return Fore.MAGENTA
            elif score >= 7:  return Fore.GREEN
            elif score >= 4:  return Fore.YELLOW
            else:             return Fore.RED

        def trend_arrow(current, previous, threshold=0.05):
            if current > previous + threshold:
                return " ↑ "
            elif current < previous - threshold:
                return " ↓ "
            else:
                return " → "

        arrow_for = {l: " → " for l in range(1, 7)}
        for episode in range(episodes):
            per_level_avg = {l: (sum(d)/len(d) if d else 0.0) for l, d in per_level_recent.items()}

            # Weight low-avg levels higher so they get more practice.
            weights = [max(0.5, 10.0 - per_level_avg[l]) for l in range(1, 7)]
            lvl = random.choices(range(1, 7), weights=weights, k=1)[0]

            env = FlappyBirdEnv(config_file_path='config.yml', show_screen=args.watch,
                                level=lvl, game_length=10)
            env.play(player=agent)

            per_level_recent[lvl].append(env.score)
            per_level_avg = {l: (sum(d)/len(d) if d else 0.0) for l, d in per_level_recent.items()}

            prev_avg_for_lvl = (per_level_avg_history[lvl][0]
                                if per_level_avg_history[lvl] else per_level_avg[lvl])
            per_level_avg_history[lvl].append(per_level_avg[lvl])
            arrow_for[lvl] = trend_arrow(per_level_avg[lvl], prev_avg_for_lvl)

            agent.episode_count += 1
            recent.append(env.score)
            running_average = sum(recent)/len(recent)

            ra_old = ra_history[0] if ra_history else running_average
            ra_history.append(running_average)

            # ----- console print --------------------------------------
            print(f"Episode {episode:4d} ", end="")
            if env.score >= 3 and env.score <= 6:
                print(Fore.YELLOW + f"| Score: {env.score}" + Style.RESET_ALL, end="")
            elif env.score >= 7 and env.score <= 9:
                print(Fore.GREEN + f"| Score: {env.score}" + Style.RESET_ALL, end="")
            elif env.score == 10:
                print(Fore.MAGENTA + f"| Score: {env.score}" + Style.RESET_ALL, end="")
            elif env.score >= 11:
                print(Fore.CYAN + f"| Score: {env.score}" + Style.RESET_ALL, end="")
            else:
                print(Fore.RED + f"| Score: {env.score}" + Style.RESET_ALL, end="")
            print(f" | Mileage: {env.mileage} | ε: {agent.epsilon:.4f} | lvl:{lvl}", end="")

            avg_summary = " ".join(
                color_for(per_level_avg[l], l) + f"Level{l}:{per_level_avg[l]:.2f}{arrow_for[l]}" + Style.RESET_ALL
                for l in range(1, 7)
            )
            print(f" | per-level avg: {avg_summary}")

            if episode % 500 == 0 and episode > 0:
                summary = " ".join(f"Level {l}:{per_level_avg[l]:.2f}" for l in range(1, 7))
                print(Fore.CYAN + f"\n--- ep {episode} | best worst-avg: {best_worst_avg:.2f} | now {summary} ---\n" + Style.RESET_ALL)

            # ----- CSV training row -----------------------------------
            if csv_train_writer is not None:
                csv_train_writer.writerow([
                    episode, lvl, env.score, env.mileage, f"{agent.epsilon:.4f}",
                    f"{per_level_avg[1]:.3f}", f"{per_level_avg[2]:.3f}",
                    f"{per_level_avg[3]:.3f}", f"{per_level_avg[4]:.3f}",
                    f"{per_level_avg[5]:.3f}", f"{per_level_avg[6]:.3f}",
                    f"{running_average:.3f}", f"{best_worst_avg:.3f}",
                ])
                csv_train_file.flush()

            # ----- live dashboard update ------------------------------
            if plotter is not None:
                h = plotter['history']
                h['episodes'].append(episode)
                h['L1'].append(per_level_avg[1])
                h['L2'].append(per_level_avg[2])
                h['L3'].append(per_level_avg[3])
                h['L4'].append(per_level_avg[4])
                h['L5'].append(per_level_avg[5])
                h['L6'].append(per_level_avg[6])
                h['epsilon'].append(agent.epsilon)
                h['running_avg'].append(running_average)
                if episode - plotter['last_drawn_ep'] >= PLOT_UPDATE_EVERY:
                    plotter['last_drawn_ep'] = episode
                    refresh_dashboard(plotter)

            # ----- epsilon decay --------------------------------------
            decay_rate = 0.99990
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= decay_rate

            # ----- in-training running-best save → sec_model_running.ckpt --
            if all(len(d) >= 20 for d in per_level_recent.values()):
                worst_avg = min(per_level_avg.values())
                if worst_avg > best_worst_avg:
                    best_worst_avg = worst_avg
                    agent.save_model(path='sec_model_running.ckpt')

            # ----- periodic honest-eval → sec_model.ckpt --------------
            if (episode + 1) % EVAL_EVERY == 0:
                eval_avg = eval_agent_on_levels(agent, range(1, 7), EVAL_EPS_PER_LEVEL)
                eval_worst = min(eval_avg.values())
                saved_sec_model = False
                if eval_worst > best_eval_worst_avg:
                    best_eval_worst_avg = eval_worst
                    agent.save_model(path='sec_model.ckpt')
                    saved_sec_model = True

                eval_summary = " ".join(f"L{l}:{eval_avg[l]:.2f}" for l in range(1, 7))
                msg = (f"[EVAL @ ep {episode+1}] {eval_summary}"
                       f" | worst:{eval_worst:.2f} | best:{best_eval_worst_avg:.2f}")
                if saved_sec_model:
                    print(Fore.MAGENTA + "\n" + msg + "  ->  SAVED sec_model.ckpt\n" + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + "\n" + msg + "\n" + Style.RESET_ALL)

                if csv_eval_writer is not None:
                    csv_eval_writer.writerow([
                        episode + 1,
                        f"{eval_avg[1]:.3f}", f"{eval_avg[2]:.3f}",
                        f"{eval_avg[3]:.3f}", f"{eval_avg[4]:.3f}",
                        f"{eval_avg[5]:.3f}", f"{eval_avg[6]:.3f}",
                        f"{eval_worst:.3f}", f"{best_eval_worst_avg:.3f}",
                        saved_sec_model,
                    ])
                    csv_eval_file.flush()

                if plotter is not None:
                    plotter['history']['eval_eps'].append(episode + 1)
                    plotter['history']['eval_worst'].append(eval_worst)
                    plotter['history']['eval_saved'].append(saved_sec_model)
                    refresh_dashboard(plotter)

            # ----- target-net refresh ---------------------------------
            if agent.episode_count % 5 == 0:
                MyAgent.update_network_model(
                    net_to_update=agent.network2,
                    net_as_source=agent.network,
                )

        # cleanup
        if csv_train_file is not None:
            csv_train_file.close()
        if csv_eval_file is not None:
            csv_eval_file.close()

        if plotter is not None:
            refresh_dashboard(plotter)
            print(Fore.CYAN + "[plot] training complete — close the dashboard "
                              "window to exit." + Style.RESET_ALL)
            plt.ioff()
            plt.show()
        raise SystemExit


    # ##############################################################
    # ##                  Single-level training                   ##
    # ##############################################################
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=args.watch,
                        level=args.level, game_length=game_length)
    agent = MyAgent(show_screen=args.watch)
    for episode in range(episodes):
        env.play(player=agent)
        agent.episode_count += 1
        recent.append(env.score)
        running_average = sum(recent)/len(recent)

        ra_old = ra_history[0] if ra_history else running_average
        ra_history.append(running_average)

        print(f"Episode {episode:4d} ", end="")
        if env.score >= 3 and env.score <= 6:
            print(Fore.YELLOW + f"| Score: {env.score}" + Style.RESET_ALL, end="")
        elif env.score >= 7 and env.score <= 9:
            print(Fore.GREEN + f"| Score: {env.score}" + Style.RESET_ALL, end="")
        elif env.score == 10:
            print(Fore.MAGENTA + f"| Score: {env.score}" + Style.RESET_ALL, end="")
        else:
            print(Fore.RED + f"| Score: {env.score}" + Style.RESET_ALL, end="")
        print(f" | Mileage: {env.mileage} | ε: {agent.epsilon:.4f} ", end="")

        if running_average >= ra_old:
            print(Fore.GREEN + f"| running_average: {running_average:.2f}" + Style.RESET_ALL)
        elif running_average >= ra_old * 0.99:
            print(Fore.YELLOW + f"| running_average: {running_average:.2f}" + Style.RESET_ALL)
        else:
            print(Fore.RED + f"| running_average: {running_average:.2f}" + Style.RESET_ALL)

        # CSV training row — 6 level-slots, only the trained level filled.
        if csv_train_writer is not None:
            level_avgs = ['', '', '', '', '', '']
            level_avgs[args.level - 1] = f"{running_average:.3f}"
            csv_train_writer.writerow([
                episode, args.level, env.score, env.mileage, f"{agent.epsilon:.4f}",
                *level_avgs,
                f"{running_average:.3f}", '',
            ])
            csv_train_file.flush()

        # Live dashboard (single-level — only the trained level moves).
        if plotter is not None:
            h = plotter['history']
            h['episodes'].append(episode)
            for i, key in enumerate(('L1', 'L2', 'L3', 'L4', 'L5', 'L6'), start=1):
                h[key].append(running_average if i == args.level else 0.0)
            h['epsilon'].append(agent.epsilon)
            h['running_avg'].append(running_average)
            if episode - plotter['last_drawn_ep'] >= PLOT_UPDATE_EVERY:
                plotter['last_drawn_ep'] = episode
                refresh_dashboard(plotter)

        decay_rate = agent.epsilon_decay
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= decay_rate

        # In-training running-best save → sec_model_running.ckpt
        if len(recent) >= 50 and running_average > best_running_average:
            best_running_average = running_average
            agent.save_model(path='sec_model_running.ckpt')

        # Periodic honest-eval → sec_model.ckpt
        if (episode + 1) % EVAL_EVERY == 0:
            eval_avg = eval_agent_on_levels(agent, [args.level], EVAL_EPS_PER_LEVEL)
            eval_worst = eval_avg[args.level]
            saved_sec_model = False
            if eval_worst > best_eval_worst_avg:
                best_eval_worst_avg = eval_worst
                agent.save_model(path='sec_model.ckpt')
                saved_sec_model = True

            msg = (f"[EVAL @ ep {episode+1}] L{args.level}:{eval_worst:.2f}"
                   f" | best:{best_eval_worst_avg:.2f}")
            if saved_sec_model:
                print(Fore.MAGENTA + "\n" + msg + "  ->  SAVED sec_model.ckpt\n" + Style.RESET_ALL)
            else:
                print(Fore.CYAN + "\n" + msg + "\n" + Style.RESET_ALL)

            if csv_eval_writer is not None:
                lvl_cells = ['', '', '', '', '', '']
                lvl_cells[args.level - 1] = f"{eval_worst:.3f}"
                csv_eval_writer.writerow([
                    episode + 1, *lvl_cells,
                    f"{eval_worst:.3f}", f"{best_eval_worst_avg:.3f}",
                    saved_sec_model,
                ])
                csv_eval_file.flush()

            if plotter is not None:
                plotter['history']['eval_eps'].append(episode + 1)
                plotter['history']['eval_worst'].append(eval_worst)
                plotter['history']['eval_saved'].append(saved_sec_model)
                refresh_dashboard(plotter)

        if agent.episode_count % 5 == 0:
            MyAgent.update_network_model(
                net_to_update=agent.network2,
                net_as_source=agent.network,
            )

    if csv_train_file is not None:
        csv_train_file.close()
    if csv_eval_file is not None:
        csv_eval_file.close()

    if plotter is not None:
        refresh_dashboard(plotter)
        print(Fore.CYAN + "[plot] training complete — close the dashboard "
                          "window to exit." + Style.RESET_ALL)
        plt.ioff()
        plt.show()

    # Final post-training sanity eval, mimicking the assignment-style harness.
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False,
                        level=args.level, game_length=game_length)
    agent2 = MyAgent(show_screen=False, load_model_path='sec_model.ckpt', mode='eval')

    episodes = 10
    scores = []
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(np.max(scores))
    print(np.mean(scores))
