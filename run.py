import logging
from decimal import Decimal

logging.basicConfig(format="%(asctime)s - %(message)s", level=20)

from agents.uniV3_pool_wealth import UniV3PoolWealthAgent
from dateutil import parser as dateparser
from policies.moving_average import MovingAveragePolicy

from dojo.environments import UniV3Env
from dojo.runners import backtest_run


def main():
    # SNIPPET 1 START
    pool = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"
    start_time = dateparser.parse("2023-04-29 10:00:00 UTC")
    end_time = dateparser.parse("2023-04-29 16:00:00 UTC")

    demo_agent = UniV3PoolWealthAgent(
        initial_portfolio={"USDC": Decimal(10_000), "WETH": Decimal(1)}
    )

    env = UniV3Env(
        date_range=(start_time, end_time),
        agents=[demo_agent],
        pools=[pool],
        backend_type="forked",
        market_impact="replay",
    )

    demo_policy = MovingAveragePolicy(
        agent=demo_agent, short_window=200, long_window=1000
    )

    sim_blocks, sim_rewards = backtest_run(env, [demo_policy], port=8051)
    # SNIPPET 1 END


if __name__ == "__main__":
    main()
