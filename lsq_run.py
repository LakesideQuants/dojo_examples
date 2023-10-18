import logging
from decimal import Decimal

logging.basicConfig(format="%(asctime)s - %(message)s", level=20)

from agents.uniV3_pool_wealth import UniV3PoolWealthAgent
from dateutil import parser as dateparser
from policies.moving_average import MovingAveragePolicy
from policies.passiveLP import PassiveConcentratedLP, Simple_LP_Deploy

from dojo.environments import UniV3Env
from dojo.runners import backtest_run
import pandas as pd
from pytz import timezone

edge_cases_configs = [
    {
        "start_time": pd.to_datetime("2023-09-24 01:44:00+00:00")
        .to_pydatetime()
        .replace(tzinfo=timezone("UTC")),
        "end_time": pd.to_datetime("2023-09-25 02:44:00+00:00")
        .to_pydatetime()
        .replace(tzinfo=timezone("UTC")),
        "deposit_token0": 5606.648922,
        "deposit_token1": 5.93962041,
        "lower_price": 0.00058026,
        "upper_price": 0.00065752,
        "token0": "USDC",
        "token1": "WETH",
        "fee_tier": 500,
        "pool_desc": "USDC/WETH-0.05",
        "description": "STF error",
    },
    {
        "start_time": pd.to_datetime("2023-09-21 15:35:00+00:00")
        .to_pydatetime()
        .replace(tzinfo=timezone("UTC")),
        "end_time": pd.to_datetime("2023-09-23 10:33:00+00:00")
        .to_pydatetime()
        .replace(tzinfo=timezone("UTC")),
        "deposit_token0": 7527.222746,
        "deposit_token1": 11.76244792,
        "lower_price": 0.00063174,
        "upper_price": 0.00063237,
        "token0": "USDC",
        "token1": "WETH",
        "fee_tier": 500,
        "pool_desc": "USDC/WETH-0.05",
        "description": "NP error",
    },
    {
        "start_time": pd.to_datetime("2023-09-16 18:46:00+00:00")
        .to_pydatetime()
        .replace(tzinfo=timezone("UTC")),
        "end_time": pd.to_datetime("2023-09-18 10:31:00+00:00")
        .to_pydatetime()
        .replace(tzinfo=timezone("UTC")),
        "deposit_token0": 11981.99022,
        "deposit_token1": 2.47269018,
        "lower_price": 0.00060636,
        "upper_price": 0.00062483,
        "token0": "USDC",
        "token1": "WETH",
        "fee_tier": 500,
        "pool_desc": "USDC/WETH-0.05",
        "description": "Result deviated from groundtruth",
    },
]


def main(edge_cases_configs):
    # SNIPPET 1 START
    for config in edge_cases_configs:
        pools = [config["pool_desc"]]
        start_time = config["start_time"]
        end_time = config["end_time"]

        # Agents
        agent1 = UniV3PoolWealthAgent(
            initial_portfolio={
                "ETH": Decimal(100),
                config["token0"]: Decimal(config["deposit_token0"]),
                config["token1"]: Decimal(config["deposit_token1"]),
            },
            name="LPAgent",
        )

        # Simulation environment (Uniswap V3)
        env = UniV3Env(
            date_range=(start_time, end_time),
            agents=[agent1],
            pools=pools,
            # backend_type="local",
            market_impact="replay",
        )

        # Policies
        passive_lp_policy = Simple_LP_Deploy(
            agent=agent1, lower_price=config["lower_price"], upper_price=config["upper_price"]
        )

        sim_blocks, sim_rewards = backtest_run(env, [passive_lp_policy], port=8051)
        # SNIPPET 1 END


if __name__ == "__main__":
    main(edge_cases_configs)
