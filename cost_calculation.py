"""
Python script to calculate Cost Per Task (CPT) for Human, Open-Source
(Self-Hosted) LLMs, and Proprietary LLMs.
This script is provided as a runnable tool.
"""

import argparse

import pandas as pd


def annualized_capex(
    initial_server_cost: float,
    n_servers: int,
    depreciation_period_years: int,
) -> float:
    """
    Calculate the annualized capital expenditure (CapEx)
    for server infrastructure.

    This represents the portion of the initial server cost amortized over the
    depreciation period, which is used to estimate the yearly cost of owning
    the hardware.

    :param initial_server_cost: Initial cost of a single server (e.g., in USD).
    :type initial_server_cost: float
    :param n_servers: Number of servers in the infrastructure.
    :type n_servers: int
    :param depreciation_period_years:
        Number of years over which the server cost is amortized.
    :type depreciation_period_years: int
    :return: Annualized CapEx (in USD).
    :rtype: float
    """
    return (initial_server_cost * n_servers) / depreciation_period_years


def power_cost_per_year(
    power_watts_per_server: float,
    hours_per_year: int,
    electricity_price_per_kwh: float,
    n_servers: int,
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
    :return: Annual power cost (in USD).
    :rtype: float
    """
    return (
        (power_watts_per_server * hours_per_year / 1000.0)
        * electricity_price_per_kwh
        * n_servers
    )


def tco_yearly(
    initial_server_cost: float,
    n_servers: int,
    depreciation_period_years: int,
    yearly_hosting_cost: float,
    yearly_software_cost: float,
    power_watts_per_server: float,
    hours_per_year: int,
    electricity_price_per_kwh: float,
    personnel_annual_cost: float,
    n_personnel: float,
    other_yearly_opex: float,
) -> float:
    """
    Calculate the Total Cost of Ownership (TCO) for the server infrastructure.

    TCO includes both capital and operational expenditures,
    amortized over time.

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
    :param personnel_annual_cost:
        Annual salary for personnel managing the infrastructure.
    :type personnel_annual_cost: float
    :param n_personnel: Fraction of an FTE amortized per server.
    :type n_personnel: float
    :param other_yearly_opex:
        Miscellaneous yearly operational expenses per server.
    :type other_yearly_opex: float
    :return: Total yearly cost of ownership (in USD).
    :rtype: float
    """
    capex_ann = annualized_capex(
        initial_server_cost, n_servers, depreciation_period_years
    )
    power = power_cost_per_year(
        power_watts_per_server,
        hours_per_year,
        electricity_price_per_kwh,
        n_servers,
    )
    personnel = personnel_annual_cost * n_personnel
    op_ex = (
        yearly_hosting_cost * n_servers
        + yearly_software_cost * n_servers
        + power
        + personnel
        + other_yearly_opex
    )
    return capex_ann + op_ex


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
    instances: float,
    uptime_fraction: float,
    initial_server_cost: float,
    n_servers: int,
    depreciation_period_years: int,
    yearly_hosting_cost: float,
    yearly_software_cost: float,
    power_watts_per_server: float,
    hours_per_year: int,
    electricity_price_per_kwh: float,
    personnel_annual_cost: float,
    n_personnel: float,
    other_yearly_opex: float,
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
    :param personnel_annual_cost:
        Annual salary for personnel managing the infrastructure.
    :type personnel_annual_cost: float
    :param n_personnel: Fraction of an FTE amortized per server.
    :type n_personnel: float
    :param other_yearly_opex:
        Miscellaneous yearly operational expenses per server.
    :type other_yearly_opex: float
    :return: The Cost Per Task (CPT) for open-source LLMs
    :rtype: float
    """
    tco = tco_yearly(
        initial_server_cost,
        n_servers,
        depreciation_period_years,
        yearly_hosting_cost,
        yearly_software_cost,
        power_watts_per_server,
        hours_per_year,
        electricity_price_per_kwh,
        personnel_annual_cost,
        n_personnel,
        other_yearly_opex,
    )
    tasks = total_yearly_tasks_from_throughput(
        rps_per_instance, instances, uptime_fraction
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
    open_cpt = cpt_open_source(
        args.rps_per_instance,
        args.instances,
        args.uptime_fraction,
        args.initial_server_cost,
        args.n_servers,
        args.depreciation_period_years,
        args.yearly_hosting_cost,
        args.yearly_software_cost,
        args.power_watts_per_server,
        args.hours_per_year,
        args.electricity_price_per_kwh,
        args.personnel_annual_cost,
        args.n_personnel,
        args.other_yearly_opex,
    )

    human_cpt = cpt_human(
        args.human_wage_gross,
        args.platform_fee_frac,
        args.qa_overhead_per_hour,
        args.time_per_task_seconds,
        args.qa_amortized_per_task,
    )

    prop_cpt = cpt_proprietary(
        args.isl_tokens,
        args.osl_tokens,
        args.price_input_per_million,
        args.price_output_per_million,
        args.api_retry_overhead_fraction,
        args.api_fixed_overhead_per_task,
    )

    qcpt_human = quality_adjusted_cpt(human_cpt, args.quality_human)
    qcpt_open = quality_adjusted_cpt(open_cpt, args.quality_open_source)
    qcpt_prop = quality_adjusted_cpt(prop_cpt, args.quality_proprietary)

    data = {
        "CPT (raw)": [human_cpt, open_cpt, prop_cpt],
        "CPT (adjusted)": [qcpt_human, qcpt_open, qcpt_prop],
        "Total cost": [
            qcpt_human * args.num_tasks,
            qcpt_open * args.num_tasks,
            qcpt_prop * args.num_tasks,
        ],
        "q": [
            args.quality_human,
            args.quality_open_source,
            args.quality_proprietary,
        ],
    }

    df = pd.DataFrame(
        data, index=["Human", "Open-Source", "Proprietary"]  # type:ignore
    )

    print(df)


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Process parameters for cost calculation."
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="The number of tasks to be executed",
    )

    # Add each parameter as a command-line argument
    parser.add_argument(
        "--initial-server-cost",
        type=float,
        default=7000.0,
        help="Initial server cost in dollars per server (8x GPUs)",
    )
    parser.add_argument(
        "--n-servers", type=int, default=1, help="Number of servers"
    )
    parser.add_argument(
        "--depreciation-period-years",
        type=int,
        default=4,
        help="Depreciation period in years",
    )
    parser.add_argument(
        "--yearly-hosting-cost",
        type=float,
        default=0.0,
        help="Yearly hosting cost in dollars per server",
    )
    parser.add_argument(
        "--yearly-software-cost",
        type=float,
        default=0.0,
        help="Yearly software cost in dollars per server",
    )
    parser.add_argument(
        "--power-watts-per-server",
        type=float,
        default=8000.0,
        help="Power consumption in Watts per server",
    )
    parser.add_argument(
        "--hours-per-year",
        type=int,
        default=24 * 365,
        help="Hours per year (default: 8760)",
    )
    parser.add_argument(
        "--electricity-price-per-kwh",
        type=float,
        default=0.12,
        help="Electricity price in dollars per kWh",
    )
    parser.add_argument(
        "--personnel-annual-cost",
        type=float,
        default=0.0,
        help="Annual salary for personnel managing infrastructure",
    )
    parser.add_argument(
        "--n-personnel",
        type=float,
        default=0.1,
        help="Fraction of an FTE amortized per server",
    )
    parser.add_argument(
        "--other-yearly-opex",
        type=float,
        default=0.0,
        help="Miscellaneous yearly OPEX per server",
    )
    parser.add_argument(
        "--rps-per-instance",
        type=float,
        default=1.0,
        help="Requests per second per instance",
    )
    parser.add_argument(
        "--instances",
        type=float,
        default=1.0,
        help="Number of instances in cluster",
    )
    parser.add_argument(
        "--uptime-fraction",
        type=float,
        default=1.0,
        help="Expected uptime / utilization fraction (0..1)",
    )
    parser.add_argument(
        "--human-wage-gross",
        type=float,
        default=12.0,
        help="Gross hourly wage for human workers (ethical benchmark)",
    )
    parser.add_argument(
        "--platform-fee-frac",
        type=float,
        default=0.333,
        help="Platform fee fraction (e.g., 33.3 percent Prolific discount)",
    )
    parser.add_argument(
        "--qa-overhead-per-hour",
        type=float,
        default=4.0,
        help="QA labor amortized per hour of data",
    )
    parser.add_argument(
        "--time-per-task-seconds",
        type=float,
        default=600.0,
        help="Estimated seconds per task",
    )
    parser.add_argument(
        "--qa-amortized-per-task",
        type=float,
        default=0.0,
        help="QA amortized per accepted task",
    )
    parser.add_argument(
        "--isl-tokens", type=int, default=50, help="Input tokens per task"
    )
    parser.add_argument(
        "--osl-tokens", type=int, default=200, help="Output tokens per task"
    )
    parser.add_argument(
        "--price-input-per-million",
        type=float,
        default=1.25,
        help="Price per 1M input tokens (e.g., GPT-5)",
    )
    parser.add_argument(
        "--price-output-per-million",
        type=float,
        default=10.00,
        help="Price per 1M output tokens (e.g., GPT-5)",
    )
    parser.add_argument(
        "--api-retry-overhead-fraction",
        type=float,
        default=0.0,
        help="Extra cost due to API retries/failures",
    )
    parser.add_argument(
        "--api-fixed-overhead-per-task",
        type=float,
        default=0.0,
        help="Fixed overhead per task (networking, marshalling)",
    )
    parser.add_argument(
        "--rps-proprietary",
        type=float,
        default=10.0,
        help="Requests per second for proprietary throughput estimation",
    )
    parser.add_argument(
        "--utilization-fraction",
        type=float,
        default=0.9,
        help="Utilization fraction (0..1)",
    )
    parser.add_argument(
        "--quality-human",
        type=float,
        default=1.0,
        help="Quality score for human workers (0..1)",
    )
    parser.add_argument(
        "--quality-open-source",
        type=float,
        default=1.0,
        help="Quality score for open-source solutions (0..1)",
    )
    parser.add_argument(
        "--quality-proprietary",
        type=float,
        default=1.0,
        help="Quality score for proprietary API (0..1)",
    )

    args = parser.parse_args()
    main(args)
