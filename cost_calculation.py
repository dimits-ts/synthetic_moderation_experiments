"""
Python script to calculate Cost Per Task (CPT) for Human, Open-Source
(Self-Hosted) LLMs, and Proprietary LLMs.
This script is provided as a runnable tool.
"""

import argparse

import pandas as pd


def annualized_capex(
    initial_server_cost, n_servers, depreciation_period_years
):
    return (initial_server_cost * n_servers) / depreciation_period_years


def power_cost_per_year(
    power_watts_per_server,
    hours_per_year,
    electricity_price_per_kwh,
    n_servers,
):
    return (
        (power_watts_per_server * hours_per_year / 1000.0)
        * electricity_price_per_kwh
        * n_servers
    )


def tco_yearly(
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
):
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
    rps_per_instance,
    instances,
    uptime_fraction,
    seconds_per_year=365 * 24 * 3600,
):
    return rps_per_instance * instances * uptime_fraction * seconds_per_year


def cpt_open_source(
    rps_per_instance,
    instances,
    uptime_fraction,
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
):
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
    if tasks <= 0:
        return float("inf"), tco, tasks
    return tco / tasks, tco, tasks


def human_cost_per_hour(w_gross, platform_fee_frac, qa_overhead_per_hour):
    return w_gross * (1 + platform_fee_frac) + qa_overhead_per_hour


def cpt_human(
    human_wage_gross,
    platform_fee_frac,
    qa_overhead_per_hour,
    time_per_task_seconds,
    qa_amortized_per_task,
):
    cph = human_cost_per_hour(
        human_wage_gross, platform_fee_frac, qa_overhead_per_hour
    )
    cpt = time_per_task_seconds / 3600.0 * cph + qa_amortized_per_task
    return cpt, cph


def cpt_proprietary(
    isl_tokens,
    osl_tokens,
    price_input_per_million,
    price_output_per_million,
    api_retry_overhead_fraction,
    api_fixed_overhead_per_task,
):
    base = (
        isl_tokens * price_input_per_million
        + osl_tokens * price_output_per_million
    ) / 1_000_000.0
    overhead = base * api_retry_overhead_fraction + api_fixed_overhead_per_task
    return base + overhead, base, overhead


def effective_llm_wage_from_tokens(
    rps_proprietary,
    utilization_fraction,
    isl_tokens,
    osl_tokens,
    price_input_per_million,
    price_output_per_million,
):
    requests_per_hour = rps_proprietary * 3600.0 * utilization_fraction
    tokens_per_hour = requests_per_hour * (isl_tokens + osl_tokens)
    cost_per_token = (
        (
            price_input_per_million * isl_tokens
            + price_output_per_million * osl_tokens
        )
        / (isl_tokens + osl_tokens)
        / 1_000_000.0
        if (isl_tokens + osl_tokens) > 0
        else 0
    )
    effective_wage = tokens_per_hour * cost_per_token
    return tokens_per_hour, effective_wage


def quality_adjusted_cpt(raw_cpt, quality_score):
    if quality_score <= 0:
        return float("inf")
    return raw_cpt / quality_score


def main(args: argparse.Namespace):
    # Open-source CPT
    open_cpt, open_tco, open_tasks = cpt_open_source(
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

    # Human CPT
    human_cpt, human_cph = cpt_human(
        args.human_wage_gross,
        args.platform_fee_frac,
        args.qa_overhead_per_hour,
        args.time_per_task_seconds,
        args.qa_amortized_per_task,
    )

    # Proprietary CPT
    prop_cpt, prop_base, prop_overhead = cpt_proprietary(
        args.isl_tokens,
        args.osl_tokens,
        args.price_input_per_million,
        args.price_output_per_million,
        args.api_retry_overhead_fraction,
        args.api_fixed_overhead_per_task,
    )

    # Effective LLM wage
    tokens_per_hour_prop, eff_wage_prop = effective_llm_wage_from_tokens(
        args.rps_proprietary,
        args.utilization_fraction,
        args.isl_tokens,
        args.osl_tokens,
        args.price_input_per_million,
        args.price_output_per_million,
    )

    # QCPT
    qcpt_human = quality_adjusted_cpt(human_cpt, args.quality_human)
    qcpt_open = quality_adjusted_cpt(open_cpt, args.quality_open_source)
    qcpt_prop = quality_adjusted_cpt(prop_cpt, args.quality_proprietary)

    # Summary
    summary = pd.DataFrame(
        [
            {
                "Resource": "Human (per task)",
                "Raw CPT ($)": human_cpt,
                "Cost Per Hour ($/h)": human_cph,
                "QCPT ($, q adj)": qcpt_human,
                "QualityScore": args.quality_human,
            },
            {
                "Resource": "Open-Source Self-Hosted (per task)",
                "Raw CPT ($)": open_cpt,
                "Yearly TCO ($)": open_tco,
                "Tasks/yr": open_tasks,
                "QCPT ($, q adj)": qcpt_open,
                "QualityScore": args.quality_open_source,
            },
            {
                "Resource": "Proprietary API (per task)",
                "Raw CPT ($)": prop_cpt,
                "Base Token Cost ($)": prop_base,
                "Overhead ($)": prop_overhead,
                "QCPT ($, q adj)": qcpt_prop,
                "QualityScore": args.quality_proprietary,
            },
        ]
    )
    print(summary)

    # Print concise results
    print("Concise results:")
    print(
        f"- Human CPT (raw): ${human_cpt:.4f} per task; effective hourly "
        f" human cost ${human_cph:.2f}/h; QCPT (q={args.quality_human}): "
        f"${qcpt_human:.4f}"
    )
    print(
        f"- Open-source CPT (raw): ${open_cpt:.6f} per task; "
        f"Yearly TCO: ${open_tco:,.2f}; Tasks/yr: {open_tasks:,.0f}; "
        f"QCPT (q={args.quality_open_source}): ${qcpt_open:.6f}"
    )
    print(
        f"- Proprietary CPT (raw): ${prop_cpt:.6f} per task "
        f"(base ${prop_base:.6f} + overhead ${prop_overhead:.6f}); "
        f"QCPT (q={args.quality_proprietary}): ${qcpt_prop:.6f}"
    )
    print(
        f"- Proprietary effective tokens/hour: {tokens_per_hour_prop:,.0f}; "
        f"effective LLM wage (equivalent $/hr by token throughput & token "
        f"prices): ${eff_wage_prop:,.2f}/hr"
    )


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Process parameters for cost calculation."
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
