# Python script to calculate Cost Per Task (CPT) for Human, Open-Source (Self-Hosted) LLMs, and Proprietary LLMs.
# This script is provided as a runnable tool. It prints a small summary table and returns a pandas DataFrame.
# It uses the formulas derived from the user's specification.
# If you want different example values, change the `params` dictionary below and re-run.

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
    # watts to kWh: watts * hours / 1000
    return (
        (power_watts_per_server * hours_per_year / 1000.0)
        * electricity_price_per_kwh
        * n_servers
    )


def tco_yearly(params):
    # sum annualized capex and opEx
    capex_ann = annualized_capex(
        params["initial_server_cost"],
        params["n_servers"],
        params["depreciation_period_years"],
    )
    power = power_cost_per_year(
        params["power_watts_per_server"],
        params["hours_per_year"],
        params["electricity_price_per_kwh"],
        params["n_servers"],
    )
    personnel = params["personnel_annual_cost"] * params["n_personnel"]
    op_ex = (
        params["yearly_hosting_cost"] * params["n_servers"]
        + params["yearly_software_cost"] * params["n_servers"]
        + power
        + personnel
        + params["other_yearly_opex"]
    )
    return capex_ann + op_ex


def total_yearly_tasks_from_throughput(
    rps_per_instance,
    instances,
    uptime_fraction,
    seconds_per_year=365 * 24 * 3600,
):
    # RPS = requests per second per instance. Multiply by number of instances 
    # and uptime to get total requests/year
    return rps_per_instance * instances * uptime_fraction * seconds_per_year


def cpt_open_source(params):
    tco = tco_yearly(params)
    tasks = total_yearly_tasks_from_throughput(
        params["rps_per_instance"],
        params["instances"],
        params["uptime_fraction"],
    )
    if tasks <= 0:
        return float("inf"), tco, tasks
    return tco / tasks, tco, tasks


def human_cost_per_hour(w_gross, platform_fee_frac, qa_overhead_per_hour):
    return w_gross * (1 + platform_fee_frac) + qa_overhead_per_hour


def cpt_human(params):
    cph = human_cost_per_hour(
        params["human_wage_gross"],
        params["platform_fee_frac"],
        params["qa_overhead_per_hour"],
    )
    cpt = (
        params["time_per_task_seconds"] / 3600.0 * cph
        + params["qa_amortized_per_task"]
    )
    return cpt, cph


def cpt_proprietary(params):
    # tokens and prices assumed per task
    base = (
        params["ISL_tokens"] * params["price_input_per_million"]
        + params["OSL_tokens"] * params["price_output_per_million"]
    ) / 1_000_000.0
    # add API overhead (retries, hidden fees) as fraction of base or as 
    # absolute per-task cost
    overhead = (
        base * params["api_retry_overhead_fraction"]
        + params["api_fixed_overhead_per_task"]
    )
    return base + overhead, base, overhead


def effective_llm_wage_from_tokens(params):
    # Tokens per hour for a given task throughput (requests/hour * tokens per 
    # request)
    requests_per_hour = (
        params["rps_proprietary"] * 3600.0 * params["utilization_fraction"]
    )
    tokens_per_hour = requests_per_hour * (
        params["ISL_tokens"] + params["OSL_tokens"]
    )
    cost_per_token = (
        (
            params["price_input_per_million"] * params["ISL_tokens"]
            + params["price_output_per_million"] * params["OSL_tokens"]
        )
        / (params["ISL_tokens"] + params["OSL_tokens"])
        / 1_000_000.0
        if (params["ISL_tokens"] + params["OSL_tokens"]) > 0
        else 0
    )
    effective_wage = tokens_per_hour * cost_per_token
    return tokens_per_hour, effective_wage


def quality_adjusted_cpt(raw_cpt, quality_score):
    # A simple QCPT: divide by quality_score (0<q<=1). Lower q increases QCPT.
    # If quality_score is zero or very small, QCPT becomes extremely large.
    if quality_score <= 0:
        return float("inf")
    return raw_cpt / quality_score


# Default example parameters taken from the user's text where available
params = {
    # Open-source / self-hosted
    "initial_server_cost": 7000.0,  # $ per server (8x GPUs)
    "n_servers": 1,
    "depreciation_period_years": 4,
    "yearly_hosting_cost": 0,  # $ per server / year
    "yearly_software_cost": 0,  # $ per server / year
    "power_watts_per_server": 8000.0,  # approx consumption in Watts for 8x 
    #GPUs under load (example)
    "hours_per_year": 24 * 365,
    "electricity_price_per_kwh": 0.12,  # $/kWh
    "personnel_annual_cost": 0.0,  # salary per person managing infra
    "n_personnel": 0.1,  # fraction of an FTE amortized per server
    "other_yearly_opex": 0.0,  # misc yearly OPEX per server
    "rps_per_instance": 1.0,  # requests per second per instance (example)
    "instances": 1.0,  # number of instances in cluster
    "uptime_fraction": 1,  # expected uptime / utilization fraction (0..1)
    # Human
    "human_wage_gross": 12.0,  # $ per hour (ethical benchmark)
    "platform_fee_frac": 0.333,  # 33.3% Prolific academic discount example
    "qa_overhead_per_hour": 4.0,  # $ per hour of data for QA labor amortized
    "time_per_task_seconds": 600.0,  # estimated seconds per task
    "qa_amortized_per_task": 0.0,  # $ extra QA amortized per accepted task
    # Proprietary API
    "ISL_tokens": 50,  # input tokens per task (example)
    "OSL_tokens": 200,  # output tokens per task (example)
    "price_input_per_million": 1.25,  # $ per 1M input tokens (GPT-5 example)
    "price_output_per_million": 10.00,  # $ per 1M output tokens (GPT-5 example)
    "api_retry_overhead_fraction": 0.0,  # 2% extra cost due to retries/failures
    "api_fixed_overhead_per_task": 0.0,  # $ fixed overhead per task (networking, marshalling)
    "rps_proprietary": 10.0,  # requests/sec for proprietary throughput estimation
    "utilization_fraction": 0.9,
    # Quality adjustment example scores (0..1)
    "quality_human": 1,
    "quality_open_source": 1,
    "quality_proprietary": 1,
}

# Compute
open_cpt, open_tco, open_tasks = cpt_open_source(params)
human_cpt, human_cph = cpt_human(params)
prop_cpt, prop_base, prop_overhead = cpt_proprietary(params)
tokens_per_hour_prop, eff_wage_prop = effective_llm_wage_from_tokens(params)
qcpt_human = quality_adjusted_cpt(human_cpt, params["quality_human"])
qcpt_open = quality_adjusted_cpt(open_cpt, params["quality_open_source"])
qcpt_prop = quality_adjusted_cpt(prop_cpt, params["quality_proprietary"])

summary = pd.DataFrame(
    [
        {
            "Resource": "Human (per task)",
            "Raw CPT ($)": human_cpt,
            "Cost Per Hour ($/h)": human_cph,
            "QCPT ($, q adj)": qcpt_human,
            "QualityScore": params["quality_human"],
        },
        {
            "Resource": "Open-Source Self-Hosted (per task)",
            "Raw CPT ($)": open_cpt,
            "Yearly TCO ($)": open_tco,
            "Tasks/yr": open_tasks,
            "QCPT ($, q adj)": qcpt_open,
            "QualityScore": params["quality_open_source"],
        },
        {
            "Resource": "Proprietary API (per task)",
            "Raw CPT ($)": prop_cpt,
            "Base Token Cost ($)": prop_base,
            "Overhead ($)": prop_overhead,
            "QCPT ($, q adj)": qcpt_prop,
            "QualityScore": params["quality_proprietary"],
        },
    ]
)

# Also print a concise text summary
print("Concise results:")
print(
    f"- Human CPT (raw): ${human_cpt:.4f} per task; effective hourly human cost ${human_cph:.2f}/h; QCPT (q={params['quality_human']}): ${qcpt_human:.4f}"
)
print(
    f"- Open-source CPT (raw): ${open_cpt:.6f} per task; Yearly TCO: ${open_tco:,.2f}; Tasks/yr: {open_tasks:,.0f}; QCPT (q={params['quality_open_source']}): ${qcpt_open:.6f}"
)
print(
    f"- Proprietary CPT (raw): ${prop_cpt:.6f} per task (base ${prop_base:.6f} + overhead ${prop_overhead:.6f}); QCPT (q={params['quality_proprietary']}): ${qcpt_prop:.6f}"
)
print(
    f"- Proprietary effective tokens/hour: {tokens_per_hour_prop:,.0f}; effective LLM wage (equivalent $/hr by token throughput & token prices): ${eff_wage_prop:,.2f}/hr"
)