# model/trading_env.py
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, feature_cols: list, initial_capital: float = 10000.0):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_capital = initial_capital
        self.n_features = len(feature_cols)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features + 3,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0.0
        self.buy_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.hold_counter = 0
        self.last_portfolio_value = self.initial_capital
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df[self.feature_cols].iloc[self.current_step].values
        price = self.df["Close"].iloc[self.current_step]
        portfolio_value = self.capital + self.position * price
        position_ratio = (self.position * price) / portfolio_value if portfolio_value > 0 else 0
        cash_ratio = self.capital / portfolio_value if portfolio_value > 0 else 1
        hold_ratio = min(self.hold_counter / 10.0, 1.0)
        obs = np.append(row, [position_ratio, cash_ratio, hold_ratio]).astype(np.float32)
        return obs

    def step(self, action):
        price = self.df["Close"].iloc[self.current_step]
        reward = 0.0

        if action == 2 and self.capital > price:  # BUY
            shares = self.capital / price
            self.position += shares
            self.capital = 0.0
            self.buy_price = price
            self.total_trades += 1
            self.hold_counter = 0
            reward = 0.1  # kleine Belohnung fürs Kaufen

        elif action == 0 and self.position > 0:  # SELL
            sell_value = self.position * price
            profit_pct = (price - self.buy_price) / self.buy_price
            # Größere Belohnung für Gewinn, größere Strafe für Verlust
            reward = profit_pct * 200
            if profit_pct > 0:
                self.winning_trades += 1
                reward += 1.0  # Bonus für gewinnenden Trade
            else:
                reward -= 1.0  # Extra Strafe für Verlust
            self.capital = sell_value
            self.position = 0.0
            self.hold_counter = 0

        elif action == 1:  # HOLD
            self.hold_counter += 1
            if self.position > 0:
                # Belohnung wenn Kurs steigt
                if self.current_step > 0:
                    prev_price = self.df["Close"].iloc[self.current_step - 1]
                    price_change = (price - prev_price) / prev_price
                    reward = price_change * 50
            else:
                # Strafe für zu langes Nichtstun
                reward = -0.05 * min(self.hold_counter, 20)

        # Portfolio Wert Änderung als zusätzliche Belohnung
        portfolio_value = self.capital + self.position * price
        portfolio_change = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        reward += portfolio_change * 10
        self.last_portfolio_value = portfolio_value

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        obs = self._get_observation()
        info = {
            "portfolio_value": portfolio_value,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades
        }

        return obs, reward, done, False, info

    def get_portfolio_value(self):
        price = self.df["Close"].iloc[self.current_step]
        return self.capital + self.position * price