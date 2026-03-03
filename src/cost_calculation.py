"""
Python script to calculate Cost Per Task (CPT) for Human, Open-Source
(Self-Hosted) LLMs, and Proprietary LLMs.
This script is provided as a runnable tool.
"""

import argparse


def capex_for_experiment(
    initial_server_cost: float,
    n_servers: int,
    depreciation_period_years: int,
    experiment_duration_days: int,
) -> float:
    """
    Calculate the annualized capital expenditure (CapEx)
    for server infrastructure only over the experiment duration.

    :param initial_server_cost: Initial cost of a single server (e.g., in USD).
    :type initial_server_cost: float
    :param n_servers: Number of servers in the infrastructure.
    :type n_servers: int
    :param depreciation_period_years:
        Number of years over which the server cost is amortized.
    :type depreciation_period_years: int
    :param experiment_duration_days:
        How many days the experiment ran for. Used to adjust capex depreciation
    :type experiment_duration_days: int
    :return: Annualized CapEx (in USD).
    :rtype: float
    """
    total_capex = initial_server_cost * n_servers
    depreciation_days = depreciation_period_years * 365
    return total_capex * (experiment_duration_days / depreciation_days)


def power_cost_for_experiment(
    power_watts_per_server: float,
    electricity_price_per_kwh: float,
    n_servers: int,
    experiment_duration_days: int,
) -> float:
    """
    Calculate the annual electricity cost for running the
    server infrastructure.

    This includes the power consumption of each server, converted to kWh,
    multiplied by the electricity price per kWh, and scaled by the number
    of servers.

    :param power_watts_per_server: Power consumption in watts per server.
    :type power_watts_per_server: float
    :param hours_per_year:
        Number of hours the server is expected to run per year.
    :type hours_per_year: int
    :param electricity_price_per_kwh:
        Price of electricity in dollars per kilowatt-hour (kWh).
    :type electricity_price_per_kwh: float
    :param n_servers: Number of servers in the infrastructure.
    :type n_servers: int
    :param experiment_duration_days:
        How many days the experiment ran for.
    :type experiment_duration_days: int
    :return: Annual power cost (in USD).
    :rtype: float
    """
    hours = experiment_duration_days * 24
    return (
        power_watts_per_server
        * hours
        / 1000.0
        * electricity_price_per_kwh
        * n_servers
    )


def tco_experiment(
    initial_server_cost: float,
    n_servers: int,
    depreciation_period_years: int,
    power_watts_per_server: float,
    electricity_price_per_kwh: float,
    experiment_duration_days: int,
) -> float:
    """
    Calculate the Total Cost of Ownership (TCO) for the server infrastructure
    for the experiment.

    TCO includes both capital and operational expenditures,
    amortized over the timeframe of the experiment.

    :param initial_server_cost: Initial cost of a single server (e.g., in USD).
    :type initial_server_cost: float
    :param n_servers: Number of servers in the infrastructure.
    :type n_servers: int
    :param depreciation_period_years:
        Number of years over which the server cost is amortized.
    :type depreciation_period_years: int
    :param power_watts_per_server: Power consumption in watts per server.
    :type power_watts_per_server: float
    :param hours_per_year:
        Number of hours the server is expected to run per year.
    :type hours_per_year: int
    :param electricity_price_per_kwh:
        Price of electricity in dollars per kilowatt-hour (kWh).
    :type electricity_price_per_kwh: float
    :param experiment_duration_days:
        How many days the experiment ran for.
    :type experiment_duration_days: int
    :return: Total yearly cost of ownership (in USD).
    :rtype: float
    """
    capex = capex_for_experiment(
        initial_server_cost,
        n_servers,
        depreciation_period_years,
        experiment_duration_days,
    )
    power = power_cost_for_experiment(
        power_watts_per_server,
        electricity_price_per_kwh,
        n_servers,
        experiment_duration_days,
    )
    return capex + power


def total_tasks_from_throughput(
    rps_per_instance: float,
    instances: int,
    uptime_fraction: float,
    experiment_duration_days: int,
    requests_per_task: int,
) -> float:
    seconds = experiment_duration_days * 24 * 3600
    total_requests = rps_per_instance * instances * uptime_fraction * seconds
    return total_requests / requests_per_task


def total_yearly_tasks_from_throughput(
    rps_per_instance: float,
    instances: float,
    uptime_fraction: float,
    seconds_per_year: int = 365 * 24 * 3600,
) -> float:
    """
    Calculate the total number of tasks handled by the system in a year.

    This is based on the system's throughput (requests per second
    per instance),
    the number of instances, and the uptime fraction (fraction of time the
    system is operational).

    :param rps_per_instance: Requests per second per instance.
    :type rps_per_instance: float
    :param instances: Number of instances in the cluster.
    :type instances: float
    :param uptime_fraction:
        Fraction of time the system is operational (e.g., 0.9 for 90% uptime).
    :type uptime_fraction: float
    :param seconds_per_year:
        Total number of seconds in a year (default: 31,536,000).
    :type seconds_per_year: int
    :return: Total number of tasks handled in a year.
    :rtype: float
    """
    return rps_per_instance * instances * uptime_fraction * seconds_per_year


def cpt_open_source(
    rps_per_instance: float,
    instances: int,
    uptime_fraction: float,
    requests_per_task: int,
    initial_server_cost: float,
    n_servers: int,
    depreciation_period_years: int,
    power_watts_per_server: float,
    electricity_price_per_kwh: float,
    experiment_duration_days: int,
) -> float:
    """
    Calculate the Cost Per Task (CPT) for self-hosted open-source LLMs.

    This function computes the CPT by normalizing the Total Cost of Ownership
    (TCO) to a per-task basis, based on the system's throughput
    and utilization.

    :param rps_per_instance: Requests per second per instance.
    :type rps_per_instance: float
    :param instances: Number of instances in the cluster.
    :type instances: float
    :param uptime_fraction:
        Fraction of time the system is operational (e.g., 0.9 for 90% uptime).
    :type uptime_fraction: float
    :param initial_server_cost: Initial cost of a single server (e.g., in USD).
    :type initial_server_cost: float
    :param n_servers: Number of servers in the infrastructure.
    :type n_servers: int
    :param depreciation_period_years:
        Number of years over which the server cost is amortized.
    :type depreciation_period_years: int
    :param yearly_hosting_cost:
        Yearly hosting cost per server (e.g., cloud hosting fees).
    :type yearly_hosting_cost: float
    :param yearly_software_cost: Yearly software licensing cost per server.
    :type yearly_software_cost: float
    :param power_watts_per_server: Power consumption in watts per server.
    :type power_watts_per_server: float
    :param hours_per_year:
        Number of hours the server is expected to run per year.
    :type hours_per_year: int
    :param electricity_price_per_kwh:
        Price of electricity in dollars per kilowatt-hour (kWh).
    :type electricity_price_per_kwh: float
    :param experiment_duration_days:
        How many days the experiment ran for.
    :type experiment_duration_days: int
    :return: The Cost Per Task (CPT) for open-source LLMs
    :rtype: float
    """
    tco = tco_experiment(
        initial_server_cost,
        n_servers,
        depreciation_period_years,
        power_watts_per_server,
        electricity_price_per_kwh,
        experiment_duration_days,
    )
    tasks = total_tasks_from_throughput(
        rps_per_instance,
        instances,
        uptime_fraction,
        experiment_duration_days,
        requests_per_task,
    )
    return tco / tasks


def human_cost_per_hour(
    human_wage_gross: float,
    platform_fee_frac: float,
    qa_overhead_per_hour: float,
) -> float:
    """
    Calculate the effective hourly cost of human labor,
    including platform fees and QA overhead.

    :param human_wage_gross: Gross hourly wage of the worker (e.g., in USD).
    :type human_wage_gross: float
    :param platform_fee_frac:
        Fraction of the wage taken by the platform (e.g., 0.33 for 33%).
    :type platform_fee_frac: float
    :param qa_overhead_per_hour:
        Additional cost per hour for quality assurance.
    :type qa_overhead_per_hour: float
    :return: Effective hourly cost of human labor (in USD).
    :rtype: float
    """
    return human_wage_gross * (1 + platform_fee_frac) + qa_overhead_per_hour


def cpt_human(
    human_wage_gross: float,
    platform_fee_frac: float,
    qa_overhead_per_hour: float,
    time_per_task_seconds: float,
    qa_amortized_per_task: float,
) -> float:
    """
    Calculate the Cost Per Task (CPT) for human labor,
    including QA amortization.

    :param human_wage_gross: Gross hourly wage of the worker (e.g., in USD).
    :type human_wage_gross: float
    :param platform_fee_frac:
        Fraction of the wage taken by the platform (e.g., 0.33 for 33%).
    :type platform_fee_frac: float
    :param qa_overhead_per_hour:
        Additional cost per hour for quality assurance.
    :type qa_overhead_per_hour: float
    :param time_per_task_seconds:
        Estimated time (in seconds) to complete a task.
    :type time_per_task_seconds: float
    :param qa_amortized_per_task:
        Amortized cost per task for quality assurance.
    :type qa_amortized_per_task: float
    :return: The human Cost Per Task (CPT).
    :rtype: float
    """
    cph = human_cost_per_hour(
        human_wage_gross, platform_fee_frac, qa_overhead_per_hour
    )
    cpt = (time_per_task_seconds / 3600.0 * cph) + qa_amortized_per_task
    return cpt


def cpt_proprietary(
    isl_tokens: float,
    osl_tokens: float,
    price_input_per_million: float,
    price_output_per_million: float,
    api_retry_overhead_fraction: float,
    api_fixed_overhead_per_task: float,
) -> float:
    """
    Calculate the Cost Per Task (CPT) for proprietary LLM APIs.

    This includes the cost of input and output tokens, plus API overhead.

    :param isl_tokens: Number of input tokens per task.
    :type isl_tokens: float
    :param osl_tokens: Number of output tokens per task.
    :type osl_tokens: float
    :param price_input_per_million: Price per million input tokens (in USD).
    :type price_input_per_million: float
    :param price_output_per_million: Price per million output tokens (in USD).
    :type price_output_per_million: float
    :param api_retry_overhead_fraction: Fraction of cost attributed to retries.
    :type api_retry_overhead_fraction: float
    :param api_fixed_overhead_per_task:
        Fixed overhead per task (e.g., API fees).
    :type api_fixed_overhead_per_task: float
    :return: The Cost Per Task (CPT) for proprietary models.
    :rtype: float
    """
    base = (
        isl_tokens * price_input_per_million
        + osl_tokens * price_output_per_million
    ) / 1_000_000.0
    overhead = base * api_retry_overhead_fraction + api_fixed_overhead_per_task
    return base + overhead


def quality_adjusted_cpt(raw_cpt: float, quality_score: float) -> float:
    """
    Adjust the Cost Per Task (CPT) based on a quality score.

    A higher quality score reduces the effective CPT, reflecting better
    data quality or efficiency.

    :param raw_cpt: Raw CPT (before quality adjustment).
    :type raw_cpt: float
    :param quality_score: Quality score (should be > 0).
    :type quality_score: float
    :return: Quality-adjusted CPT.
    :rtype: float
    """
    if quality_score <= 0:
        return float("inf")
    return raw_cpt / quality_score


def main(args: argparse.Namespace):

    match args.mode:
        case "open-source":
            cpt = cpt_open_source(
                args.rps_per_instance,
                args.instances,
                args.uptime_fraction,
                args.requests_per_task,
                args.initial_server_cost,
                args.n_servers,
                args.depreciation_period_years,
                args.power_watts_per_server,
                args.electricity_price_per_kwh,
                args.experiment_duration_days,
            )

        case "human":
            cpt = cpt_human(
                args.human_wage_gross,
                args.platform_fee_frac,
                args.qa_overhead_per_hour,
                args.time_per_task_seconds,
                args.qa_amortized_per_task,
            )

        case "proprietary":
            cpt = cpt_proprietary(
                args.isl_tokens,
                args.osl_tokens,
                args.price_input_per_million,
                args.price_output_per_million,
                args.api_retry_overhead_fraction,
                args.api_fixed_overhead_per_task,
            )

        case _:
            raise ValueError(f"Unknown mode: {args.mode}")

    print(f"CPT ({args.mode}): {cpt:.6f} USD")

    if args.num_tasks > 1:
        print(
            f"Total cost ({args.num_tasks} tasks): {cpt * args.num_tasks:.2f} USD"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=["open-source", "human", "proprietary"],
        required=True,
        help="Which cost model to compute",
    )

    # Experiment structure
    parser.add_argument("--requests-per-task", type=int, default=1)
    parser.add_argument("--experiment-duration-days", type=int, default=1)

    # Open-source infra
    parser.add_argument("--initial-server-cost", type=float, default=7000.0)
    parser.add_argument("--n-servers", type=int, default=1)
    parser.add_argument("--depreciation-period-years", type=int, default=4)
    parser.add_argument("--power-watts-per-server", type=float, default=8000.0)
    parser.add_argument(
        "--electricity-price-per-kwh", type=float, default=0.12
    )
    parser.add_argument("--rps-per-instance", type=float, default=0.2)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--uptime-fraction", type=float, default=0.9)

    # Human labor
    parser.add_argument("--human-wage-gross", type=float, default=12.0)
    parser.add_argument("--platform-fee-frac", type=float, default=0.333)
    parser.add_argument("--qa-overhead-per-hour", type=float, default=4.0)
    parser.add_argument("--time-per-task-seconds", type=float, default=600.0)
    parser.add_argument("--qa-amortized-per-task", type=float, default=0.0)

    # Proprietary (full discussion defaults)
    parser.add_argument("--isl-tokens", type=int, default=50)
    parser.add_argument("--osl-tokens", type=int, default=20)
    parser.add_argument("--price-input-per-million", type=float, default=1.25)
    parser.add_argument("--price-output-per-million", type=float, default=10.0)
    parser.add_argument(
        "--api-retry-overhead-fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--api-fixed-overhead-per-task", type=float, default=0.0
    )

    main(parser.parse_args())
