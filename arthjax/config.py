"""Economy-wide parameters (single source of truth)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EconomyConfig:
    """Default scale and structure for the ArthJAX synthetic economy."""

    num_households: int = 250
    num_companies: int = 60
    num_sectors: int = 10
    num_commodities: int = 6
    num_currencies: int = 4
    num_agent_types: int = 4  # value, momentum, panic, saver
    contagion_steps: int = 3

    # Simulation defaults
    default_num_steps: int = 600
    default_seed: int = 42

    # Macro anchors
    initial_interest_rate: float = 0.04
    initial_inflation: float = 0.02
    initial_unemployment: float = 0.05
    initial_sentiment: float = 0.5
    nairu: float = 0.05

    # init_state defaults
    household_wealth_base: float = 100.0
    household_wealth_noise: float = 20.0
    household_wealth_min: float = 20.0
    household_wealth_max: float = 500.0
    household_income_base: float = 5.0
    household_income_noise: float = 1.0
    household_income_min: float = 1.0
    household_income_max: float = 20.0
    household_risk_by_type: tuple = (0.3, 0.7, 0.8, 0.1)
    company_cash_base: float = 500.0
    company_cash_noise: float = 100.0
    company_debt_base: float = 200.0
    company_debt_noise: float = 50.0
    company_stock_price_base: float = 100.0
    company_stock_price_noise: float = 20.0
    company_initial_revenue: float = 6.5
    bank_deposits_init: float = 150000.0
    bank_loans_init: float = 60000.0
    bank_bad_loans_init: float = 5000.0
    bank_leverage_init: float = 8.0
    commodity_price_base: float = 50.0
    stock_volatility_init: float = 0.15

    # Household dynamics
    income_sentiment_base: float = 0.7
    income_sentiment_scale: float = 0.6
    consumption_rates: tuple = (0.4, 0.5, 0.6, 0.3)
    wealth_clip_min: float = 15.0
    wealth_clip_max: float = 800.0
    total_demand_min: float = 80.0
    total_demand_max: float = 1200.0

    # Company dynamics
    leverage_threshold: float = 0.8
    log_price_min: float = 55.0
    log_price_max: float = 175.0

    # Banking
    savings_rate: float = 0.15
    default_cash_threshold: float = -1000.0

    # Markets
    log_anchor_price: float = 100.0
    market_revert_rate: float = 0.035
    vol_decay: float = 0.82
    vol_min: float = 0.04
    vol_max: float = 0.28

    # Shocks
    shock_prob: float = 0.025
    shock_magnitude_min: float = 0.1
    shock_magnitude_max: float = 0.5
    oil_index_min: float = 0.5
    oil_index_max: float = 3.0

    # Contagion
    contagion_momentum: float = 0.6
    contagion_propagation: float = 0.4
    contagion_mean_revert: float = 0.95

    # Macro feedback
    growth_base: float = 0.025
    cycle_freq_1: float = 0.04
    cycle_freq_2: float = 0.013
    unemployment_min: float = 0.025
    unemployment_max: float = 0.11
    inflation_target: float = 0.025
    taylor_inflation_coef: float = 1.0
    taylor_unemployment_coef: float = -0.35
    taylor_rate_step: float = 0.012
    rate_min: float = 0.02
    rate_max: float = 0.09

    # Metrics
    gdp_min: float = 200.0
    gdp_max: float = 2000.0
    stock_index_max: float = 10000.0

    # World model defaults
    world_model_hidden_dim: int = 128
    world_model_epochs: int = 80
    world_model_batch_size: int = 64
    world_model_lr: float = 0.003
    world_model_num_rollouts: int = 8
    world_model_rollout_length: int = 150
    world_model_eval_steps: int = 200
    world_model_grad_clip: float = 1.0
    world_model_norm_clip: float = 4.0
    world_model_macro_only: bool = True
    world_model_latent_dim: int = 32
    world_model_eval_seed: int = 200
    world_model_use_latent: bool = False
    world_model_multi_step_horizon: int = 3

    # Visualization
    plot_burn_in: int = 25


DEFAULT_CONFIG = EconomyConfig()
