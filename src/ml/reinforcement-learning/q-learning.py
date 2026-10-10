import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.config import CLEAN_DIR, CUTOFF

FLAT = 0
LONG = 1

BUY = 0
SELL = 1
HOLD = 2

EDGES = np.array([0.95, 0.98, 1.02, 1.05])
MARKET_BINS = len(EDGES) + 1
WINDOW = 20
DISPLAY_PLOTS = True

ACTION_NAME = {BUY: "buy", SELL: "sell", HOLD: "hold"}
POSITION_NAME = {FLAT: "flat", LONG: "long"}


class QLearning:
    def __init__(self, df: pd.DataFrame):
        """
        Allocate a Q-table of zeros. Rows are discrete state ids. Columns are
        action ids (BUY, SELL, HOLD). The number of rows is the number of
        (position, market-bin) pairs you can actually be in.
        """
        self.df = df
        self.alpha = 0.1
        self.gamma = 0.9
        self.epochs = 1000
        self.exploration_prob = 0.1
        self.q_table = np.zeros((2 * MARKET_BINS, 3))

        lagged = df["Adj Close"].shift(1)
        rolling_mean = lagged.rolling(window=WINDOW).mean()
        df["Price_to_SMA_Ratio"] = (df["Adj Close"] / rolling_mean)

    def market_bin(self, t: int):
        if t < WINDOW:
            return -1
        else:
            return int(np.digitize(self.df["Price_to_SMA_Ratio"].iloc[t], EDGES))

    def state_id(self, position: int, market_bin: int) -> int:
        return position * MARKET_BINS + market_bin

    def legal_actions(self, position: int) -> list[int]:
        if position == FLAT:
            return [BUY, HOLD]
        elif position == LONG:
            return [HOLD, SELL]

    def select_action(self, state_id: int, position: int) -> int:
        legal_actions = self.legal_actions(position)
        if np.random.rand() < self.exploration_prob:
            return int(np.random.choice(legal_actions))
        values = [self.q_table[state_id, action] for action in legal_actions]
        return legal_actions[int(np.argmax(values))]

    def get_initial_state(self):
        return self.state_id(FLAT, self.market_bin(WINDOW)), FLAT, WINDOW

    def _update_state(self, position: int, action: int) -> int:
        if action == HOLD:
            return position
        if position == FLAT and action == BUY:
            return LONG
        if position == LONG and action == SELL:
            return FLAT
        raise ValueError(f"illegal action {action} from position {position}")

    def get_next_state(self, t: int, position: int, action: int):
        next_t = t + 1
        if next_t >= len(self.df):
            return None

        next_position = self._update_state(position, action)
        next_id = self.state_id(next_position, self.market_bin(next_t))
        return next_id, next_position, next_t

    def calculate_reward(self, position: int, t: int) -> float:
        if position != LONG:
            return 0.0
        price = self.df["Adj Close"].iloc[t]
        next_price = self.df["Adj Close"].iloc[t + 1]
        return float(next_price / price - 1.0)

    def update(
        self,
        state_id: int,
        action: int,
        reward: float,
        next_state_id: int,
        next_position: int,
        done: bool,
    ):
        if done:
            target = reward
        else:
            legal_moves = self.legal_actions(next_position)
            future = max(self.q_table[next_state_id, action] for action in legal_moves)
            target = reward + self.gamma * future
        current = self.q_table[state_id, action]
        self.q_table[state_id, action] = (1 - self.alpha) * current + self.alpha * target

    def _dates(self) -> np.ndarray:
        return pd.to_datetime(self.df["Date"]).to_numpy()

    def train(self):
        dates = self._dates()
        cutoff = np.datetime64(CUTOFF)
        train_idx = np.flatnonzero(dates < cutoff)
        if len(train_idx) == 0:
            raise ValueError(f"no bars before {CUTOFF}")
        last_train = int(train_idx[-1])

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
                self.update(
                    state_id,
                    action,
                    reward,
                    next_state_id,
                    next_position,
                    done,
                )
                state_id, position, t = next_state_id, next_position, next_t
                if done:
                    break
            filled = int(bar_width * (epoch + 1) / self.epochs)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(f"\r[{bar}] {epoch + 1}/{self.epochs}", end="", flush=True)
        print()

    def predict(self, t: int, position: int) -> int:
        state_id = self.state_id(position, self.market_bin(t))
        legal = self.legal_actions(position)
        values = [self.q_table[state_id, action] for action in legal]
        return legal[int(np.argmax(values))]

    def evaluate(self) -> float:
        test_idx = np.flatnonzero(self._dates() >= np.datetime64(CUTOFF))

        t = int(test_idx[0])
        if self.market_bin(t) < 0:
            raise ValueError("test window starts before the market signal exists")

        position = FLAT
        total = 0.0
        steps = 0
        while True:
            action = self.predict(t, position)
            step = self.get_next_state(t, position, action)
            if step is None:
                break
            _, next_position, next_t = step
            total += self.calculate_reward(next_position, t)

            position, t = next_position, next_t
            steps += 1

        first = int(test_idx[0])
        last = int(test_idx[-1])
        held = float(self.df["Adj Close"].iloc[last] / self.df["Adj Close"].iloc[first] - 1.0)
        print(f"Evaluate steps: {steps}")
        print(f"Policy equity change: {total:.4f}")
        print(f"Buy-and-hold equity change: {held:.4f}")
        return total

    def _split_bounds(self) -> tuple[int, int]:
        dates = self._dates()
        cutoff = np.datetime64(CUTOFF)
        train_idx = np.flatnonzero(dates < cutoff)
        test_idx = np.flatnonzero(dates >= cutoff)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"cutoff {CUTOFF} does not split this frame")
        return int(train_idx[-1]), int(test_idx[0])

    def _greedy_path(self, start: int, last: int, equity: float) -> tuple[list[int], list[float], float]:
        indexes = [start]
        equities = [equity]
        position = FLAT
        t = start
        while True:
            action = self.predict(t, position)
            step = self.get_next_state(t, position, action)
            if step is None or int(step[2]) > last:
                break
            _, next_position, next_t = step
            equity *= 1.0 + self.calculate_reward(next_position, t)
            indexes.append(next_t)
            equities.append(equity)
            position = next_position
            t = next_t
        return indexes, equities, equity

    def _ratio_label(self, bin_id: int) -> str:
        if bin_id <= 0:
            return f"< {EDGES[0]:.2f}"
        if bin_id >= len(EDGES):
            return f">= {EDGES[-1]:.2f}"
        return f"{EDGES[bin_id - 1]:.2f} to {EDGES[bin_id]:.2f}"

    def _policy_table(self) -> pd.DataFrame:
        rows = []
        for position in (FLAT, LONG):
            legal = set(self.legal_actions(position))
            for bin_id in range(MARKET_BINS):
                state_id = self.state_id(position, bin_id)
                greedy = max(legal, key=lambda action: self.q_table[state_id, action])
                row = {
                    "position": POSITION_NAME[position],
                    "bin": bin_id,
                    "price / prior SMA": self._ratio_label(bin_id),
                    "policy": ACTION_NAME[greedy],
                }
                for action in (BUY, HOLD, SELL):
                    if action in legal:
                        row[ACTION_NAME[action]] = f"{self.q_table[state_id, action]:.4f}"
                    else:
                        row[ACTION_NAME[action]] = ""
                rows.append(row)
        return pd.DataFrame(rows)[
            ["position", "bin", "price / prior SMA", "buy", "hold", "sell", "policy"]
        ]

    def visualize(self, symbol: str) -> None:
        """
        Greedy growth of $1 against the stock, train then test.
        The test walk starts flat, same as evaluate. Prints the greedy policy.
        """
        last_train, first_test = self._split_bounds()
        train_idx, train_eq, end_eq = self._greedy_path(WINDOW, last_train, 1.0)
        test_idx, test_eq, _ = self._greedy_path(first_test, len(self.df) - 1, end_eq)

        indexes = np.asarray(train_idx + test_idx)
        policy = np.asarray(train_eq + test_eq, dtype=float)
        prices = self.df["Adj Close"].to_numpy()
        dates = pd.to_datetime(self._dates())
        stock = prices[indexes] / prices[indexes[0]]

        train_policy = train_eq[-1] / train_eq[0] - 1.0
        test_policy = test_eq[-1] / test_eq[0] - 1.0
        train_stock = float(prices[last_train] / prices[WINDOW] - 1.0)
        test_stock = float(prices[-1] / prices[first_test] - 1.0)
        print(self._policy_table().to_string(index=False))
        print(f"Train growth of $1: policy {train_policy:.4f}  stock {train_stock:.4f}")
        print(f"Test growth of $1:  policy {test_policy:.4f}  stock {test_stock:.4f}")

        if not DISPLAY_PLOTS:
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axvspan(dates[first_test], dates[indexes[-1]], color="0.85", label="test", zorder=0)
        ax.plot(dates[indexes], stock, label="stock", color="tab:blue")
        ax.plot(dates[indexes], policy, label="policy", color="tab:orange")
        ax.axvline(pd.Timestamp(CUTOFF), color="0.4", linestyle="--", linewidth=1)
        ax.set_title(f"{symbol} Q-learning")
        ax.set_ylabel("growth of $1")
        ax.set_xlabel("Date")
        ax.legend()
        fig.tight_layout()
        plt.show()


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = pd.read_csv(CLEAN_DIR / f"{symbol}.csv", parse_dates=["Date"])
    df = df.sort_values("Date").dropna().reset_index(drop=True)

    agent = QLearning(df)
    print(f"Training {symbol} for {agent.epochs} epochs, cutoff {CUTOFF}")
    agent.train()
    agent.evaluate()
    agent.visualize(symbol)


if __name__ == "__main__":
    main()