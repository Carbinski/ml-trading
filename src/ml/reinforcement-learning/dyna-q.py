import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.config import CLEAN_DIR, CUTOFF

_SPEC = importlib.util.spec_from_file_location(
    "q_learning_agent", Path(__file__).with_name("q-learning.py")
)
q_learning = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(q_learning)

SEED = q_learning.SEED
EPOCHS = q_learning.EPOCHS
ALPHAS = q_learning.ALPHAS
DISPLAY_PLOTS = q_learning.DISPLAY_PLOTS
PLANNING_STEPS = 5


class DynaQ(q_learning.QLearning):
    def __init__(
        self,
        df: pd.DataFrame,
        alpha: float,
        exploration_prob: float = 0.1,
        gamma: float | None = None,
        epochs: int | None = None,
        planning_steps: int | None = None,
    ):
        super().__init__(
            df,
            alpha,
            exploration_prob=exploration_prob,
            gamma=gamma,
            epochs=epochs,
        )
        self.planning_steps = (PLANNING_STEPS if planning_steps is None else int(planning_steps))
        self.init_model()

    def init_model(self) -> None:
        self.model = {}

    def record_transition(
        self,
        state_id: int,
        action: int,
        reward: float,
        next_state_id: int,
        next_position: int,
        done: bool,
    ) -> None:
        key = (state_id, action)
        value = (reward, next_state_id, next_position, done)
        self.model[key] = value

    def sample_transition(self) -> tuple[int, int, float, int, int, bool] | None:
        if not self.model:
            return None

        states = list({state_id for state_id, _action in self.model})
        state_id = states[int(np.random.randint(len(states)))]
        actions = [action for s, action in self.model if s == state_id]
        action = actions[int(np.random.randint(len(actions)))]
        reward, next_state_id, next_position, done = self.model[(state_id, action)]
        return state_id, action, reward, next_state_id, next_position, done

    def plan(self) -> None:
        for _ in range(self.planning_steps):
            transition = self.sample_transition()
            if transition is None:
                return
            self.update(*transition)

    def train(self, seed: int = SEED) -> None:
        """Train on the history, and rehearse after every real backup.

        After that update, including the transition that ends the window,
        record_transition with the same arguments, then plan. Then advance
        to the next state. End the epoch when done.

        Keep the model from one epoch to the next. The progress bar can match
        Q-learning.
        """
        dates, cutoff, train_idx, last_train = self._initialize_training(seed, CUTOFF)

        bar_width = 30
        for epoch in range(self.epochs):
            state_id, position, t = self.get_initial_state()
            if t > last_train:
                break
            while True:
                action = self.select_action(state_id, position)
                step = self.get_next_state(t, position, action)
                if step is None or int(step[2]) > last_train:
                    break
                next_state_id, next_position, next_t = step
                reward = self.calculate_reward(next_position, t)
                done = next_t == last_train
                self.update(state_id, action, reward, next_state_id, next_position, done)
                self.record_transition(state_id, action, reward, next_state_id, next_position, done)
                self.plan()
                state_id, position, t = next_state_id, next_position, next_t
                if done:
                    break
            filled = int(bar_width * (epoch + 1) / self.epochs)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(f"\r[{bar}] {epoch + 1}/{self.epochs}", end="", flush=True)
        print()


def plot_curves(symbol: str, curves: list[dict]) -> None:
    if not DISPLAY_PLOTS or not curves:
        return
    base = curves[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axvspan(base["first_test"], base["last"], color="0.85", label="test", zorder=0)
    ax.plot(base["dates"], base["stock"], label="stock", color="tab:blue")
    for curve in curves:
        ax.plot(curve["dates"], curve["policy"], label=f"alpha={curve['alpha']:g}")
    ax.axvline(pd.Timestamp(CUTOFF), color="0.4", linestyle="--", linewidth=1)
    ax.set_title(f"{symbol} Dyna-Q")
    ax.set_ylabel("growth of $1")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    plt.show()


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = pd.read_csv(CLEAN_DIR / f"{symbol}.csv", parse_dates=["Date"])
    df = df.sort_values("Date").dropna().reset_index(drop=True)
    q_learning.add_price_ratio(df)
    q_learning.print_bin_counts(df)

    curves = []
    rows = []
    for alpha in ALPHAS:
        agent = DynaQ(df, alpha)
        print(
            f"\nTraining {symbol} for {agent.epochs} epochs, "
            f"alpha={agent.alpha:g}, planning_steps={agent.planning_steps}, "
            f"cutoff {CUTOFF}"
        )
        agent.train()
        agent.evaluate()
        curve = agent.visualize(symbol)
        rows.append(
            {
                "alpha": curve["alpha"],
                "train": curve["train_policy"],
                "test": curve["test_policy"],
            }
        )
        curves.append(curve)

    print(
        f"\nStock growth of $1: train {curves[0]['train_stock']:.4f}  "
        f"test {curves[0]['test_stock']:.4f}"
    )
    print("Alpha grid")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    plot_curves(symbol, curves)


if __name__ == "__main__":
    main()
