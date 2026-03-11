# Intervention Detection in Discussions
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

"""
Python script to calculate Cost Per Task (CPT) for Human, Open-Source
(Self-Hosted) LLMs, and Proprietary LLMs, following the methodology defined
in the paper.

Total Cost = CPT * N_tasks
CPT = O_task * C_op

Three cost models are supported:
  - open-source:  Self-hosted LLMs; C_op derived from amortized TCO.
  - proprietary:  API-based LLMs; C_op derived from token usage.
  - human:        Crowdsourced workers; C_op derived from wage + platform fees.
"""

import argparse


# ---------------------------------------------------------------------------
# Open-source (self-hosted) model
# ---------------------------------------------------------------------------


def capex_for_experiment(
    server_cost: float,
    n_servers: int,
    depreciation_years: int,
    experiment_days: int,
) -> float:
    """
    Amortized capital expenditure over the experiment window.

    CapEx_exp = C_server * N_servers * (D_exp / (365 * Y_depr))

    :param server_cost: Purchase cost of a single server (USD).
    :param n_servers: Number of servers.
    :param depreciation_years: Hardware depreciation period (years).
    :param experiment_days: Duration of the experiment (days).
    :return: Amortized CapEx for the experiment window (USD).
    """
    return (
        server_cost
        * n_servers
        * (experiment_days / (365 * depreciation_years))
    )


def power_cost_for_experiment(
    power_watts: float,
    electricity_price_per_kwh: float,
    n_servers: int,
    experiment_days: int,
) -> float:
    """
    Electricity cost over the experiment window.

    PowerCost_exp = (P_watts / 1000) * (24 * D_exp) * C_kWh * N_servers

    :param power_watts: Average power draw per server (W).
    :param electricity_price_per_kwh: Electricity price (USD/kWh).
    :param n_servers: Number of servers.
    :param experiment_days: Duration of the experiment (days).
    :return: Total electricity cost for the experiment window (USD).
    """
    return (
        (power_watts / 1000.0)
        * (24 * experiment_days)
        * electricity_price_per_kwh
        * n_servers
    )


def tco_experiment(
    server_cost: float,
    n_servers: int,
    depreciation_years: int,
    power_watts: float,
    electricity_price_per_kwh: float,
    experiment_days: int,
) -> float:
    """
    Total Cost of Ownership for the experiment window.

    TCO_exp = CapEx_exp + PowerCost_exp

    :param server_cost: Purchase cost of a single server (USD).
    :param n_servers: Number of servers.
    :param depreciation_years: Hardware depreciation period (years).
    :param power_watts: Average power draw per server (W).
    :param electricity_price_per_kwh: Electricity price (USD/kWh).
    :param experiment_days: Duration of the experiment (days).
    :return: TCO for the experiment window (USD).
    """
    capex = capex_for_experiment(
        server_cost, n_servers, depreciation_years, experiment_days
    )
    power = power_cost_for_experiment(
        power_watts, electricity_price_per_kwh, n_servers, experiment_days
    )
    return capex + power


def total_requests_for_experiment(
    rps_per_instance: float,
    n_instances: int,
    utilization: float,
    experiment_days: int,
) -> float:
    """
    Total number of model requests processed during the experiment.

    Requests_exp = R_ps * N_inst * U * (24 * 3600 * D_exp)

    :param rps_per_instance: Request throughput per model instance (req/s).
    :param n_instances: Number of model instances.
    :param utilization: GPU utilization fraction (0–1).
    :param experiment_days: Duration of the experiment (days).
    :return: Total number of requests processed.
    """
    return (
        rps_per_instance
        * n_instances
        * utilization
        * (24 * 3600 * experiment_days)
    )


def cpt_open_source(
    # Infrastructure
    os_server_cost: float,
    os_n_servers: int,
    os_depreciation_years: int,
    os_power_watts: float,
    os_electricity_price_per_kwh: float,
    # Throughput
    os_rps_per_instance: float,
    os_n_instances: int,
    os_utilization: float,
    # Task structure
    experiment_days: int,
    operations_per_task: int,
) -> float:
    """
    Cost Per Task for self-hosted open-source LLMs.

    C_op^OS  = TCO_exp / Requests_exp
    CPT_OS   = O_task * C_op^OS

    :param os_server_cost: Purchase cost of a single server (USD).
    :param os_n_servers: Number of servers.
    :param os_depreciation_years: Hardware depreciation period (years).
    :param os_power_watts: Average power draw per server (W).
    :param os_electricity_price_per_kwh: Electricity price (USD/kWh).
    :param os_rps_per_instance: Request throughput per model instance (req/s).
    :param os_n_instances: Number of model instances.
    :param os_utilization: GPU utilization fraction (0–1).
    :param experiment_days: Duration of the experiment (days).
    :param operations_per_task: Number of model requests per task (O_task).
    :return: Cost Per Task (USD).
    """
    tco = tco_experiment(
        os_server_cost,
        os_n_servers,
        os_depreciation_years,
        os_power_watts,
        os_electricity_price_per_kwh,
        experiment_days,
    )
    requests = total_requests_for_experiment(
        os_rps_per_instance,
        os_n_instances,
        os_utilization,
        experiment_days,
    )
    c_op = tco / requests
    return operations_per_task * c_op


# ---------------------------------------------------------------------------
# Proprietary model
# ---------------------------------------------------------------------------


def cpt_proprietary(
    prop_isl: int,
    prop_osl: int,
    prop_price_in_per_million: float,
    prop_price_out_per_million: float,
    prop_api_overhead: float,
    operations_per_task: int,
) -> float:
    """
    Cost Per Task for proprietary LLM APIs.

    C_op^Prop = (ISL * P_in + OSL * P_out) / 1_000_000 + C_API_Overhead
    CPT_Prop  = O_task * C_op^Prop

    :param prop_isl: Input sequence length (tokens) per model call.
    :param prop_osl: Output sequence length (tokens) per model call.
    :param prop_price_in_per_million: Vendor price per million input tokens (USD).
    :param prop_price_out_per_million: Vendor price per million output tokens (USD).
    :param prop_api_overhead: Fixed API overhead cost per call (USD).
    :param operations_per_task: Number of model calls per task (O_task).
    :return: Cost Per Task (USD).
    """
    c_op = (
        prop_isl * prop_price_in_per_million
        + prop_osl * prop_price_out_per_million
    ) / 1_000_000.0 + prop_api_overhead
    return operations_per_task * c_op


# ---------------------------------------------------------------------------
# Human model
# ---------------------------------------------------------------------------


def cpt_human(
    human_wage_gross: float,
    human_platform_fee_frac: float,
    human_time_per_task_seconds: float,
    human_qa_amortized_per_task: float,
    human_n_humans: int,
) -> float:
    """
    Cost Per Task for human participants.

    C_op^Human = (W_gross / 3600) * (1 + F_platform) * T_task + C_QA
    CPT_Human  = C_op^Human * N_Humans

    :param human_wage_gross: Ethical gross hourly wage of a worker (USD/hr).
    :param human_platform_fee_frac: Platform commission as a fraction (e.g. 0.20).
    :param human_time_per_task_seconds: Estimated task duration per worker (s).
    :param human_qa_amortized_per_task: Amortized QA overhead cost per task (USD).
    :param human_n_humans: Number of human workers needed to complete one task.
    :return: Cost Per Task (USD).
    """
    c_op = (human_wage_gross / 3600.0) * (
        1 + human_platform_fee_frac
    ) * human_time_per_task_seconds + human_qa_amortized_per_task
    return c_op * human_n_humans


# ---------------------------------------------------------------------------
# Quality adjustment (optional)
# ---------------------------------------------------------------------------


def quality_adjusted_cpt(raw_cpt: float, quality_score: float) -> float:
    """
    Adjust the CPT by a quality score.

    A higher quality score lowers the effective CPT.

    :param raw_cpt: Raw CPT before quality adjustment (USD).
    :param quality_score: Quality score (must be > 0).
    :return: Quality-adjusted CPT (USD).
    """
    if quality_score <= 0:
        return float("inf")
    return raw_cpt / quality_score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace):
    match args.mode:
        case "open-source":
            cpt = cpt_open_source(
                os_server_cost=args.os_server_cost,
                os_n_servers=args.os_n_servers,
                os_depreciation_years=args.os_depreciation_years,
                os_power_watts=args.os_power_watts,
                os_electricity_price_per_kwh=args.os_electricity_price_per_kwh,
                os_rps_per_instance=args.os_rps_per_instance,
                os_n_instances=args.os_n_instances,
                os_utilization=args.os_utilization,
                experiment_days=args.experiment_days,
                operations_per_task=args.operations_per_task,
            )

        case "proprietary":
            cpt = cpt_proprietary(
                prop_isl=args.prop_isl,
                prop_osl=args.prop_osl,
                prop_price_in_per_million=args.prop_price_in_per_million,
                prop_price_out_per_million=args.prop_price_out_per_million,
                prop_api_overhead=args.prop_api_overhead,
                operations_per_task=args.operations_per_task,
            )

        case "human":
            cpt = cpt_human(
                human_wage_gross=args.human_wage_gross,
                human_platform_fee_frac=args.human_platform_fee_frac,
                human_time_per_task_seconds=args.human_time_per_task_seconds,
                human_qa_amortized_per_task=args.human_qa_amortized_per_task,
                human_n_humans=args.human_n_humans,
            )

        case _:
            raise ValueError(f"Unknown mode: {args.mode}")

    if args.quality_score is not None:
        adjusted = quality_adjusted_cpt(cpt, args.quality_score)
        print(
            f"CPT ({args.mode}): {cpt:.6f} USD  |  quality-adjusted: {adjusted:.6f} USD"
        )
    else:
        print(f"CPT ({args.mode}): {cpt:.6f} USD")

    if args.n_tasks > 1:
        print(
            f"Total cost ({args.n_tasks} tasks): {cpt * args.n_tasks:.2f} USD"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Cost Per Task (CPT) for open-source LLMs, proprietary LLMs, or human workers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Shared ----
    parser.add_argument(
        "--mode",
        choices=["open-source", "proprietary", "human"],
        required=True,
        help="Cost model to compute.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=1,
        help="Total number of tasks (N_tasks). Used to report total cost.",
    )
    parser.add_argument(
        "--operations-per-task",
        type=int,
        default=1,
        help="Number of model calls / operations required per task (O_task). "
        "Used by open-source and proprietary modes.",
    )
    parser.add_argument(
        "--experiment-days",
        type=int,
        default=1,
        help="Duration of the experiment in days (D_exp). Used by open-source mode.",
    )
    parser.add_argument(
        "--quality-score",
        type=float,
        default=None,
        help="Optional quality score (>0) to compute quality-adjusted CPT.",
    )

    # ---- Open-source: infrastructure ----
    os_group = parser.add_argument_group("open-source — infrastructure")
    os_group.add_argument(
        "--os-server-cost",
        type=float,
        default=7_000.0,
        help="Purchase cost of a single server, C_server (USD).",
    )
    os_group.add_argument(
        "--os-n-servers",
        type=int,
        default=1,
        help="Number of servers, N_servers.",
    )
    os_group.add_argument(
        "--os-depreciation-years",
        type=int,
        default=4,
        help="Hardware depreciation period, Y_depr (years).",
    )
    os_group.add_argument(
        "--os-power-watts",
        type=float,
        default=500.0,
        help="Average server power draw, P_watts (W).",
    )
    os_group.add_argument(
        "--os-electricity-price-per-kwh",
        type=float,
        default=0.12,
        help="Electricity price, C_kWh (USD/kWh).",
    )

    # ---- Open-source: throughput ----
    os_tp_group = parser.add_argument_group("open-source — throughput")
    os_tp_group.add_argument(
        "--os-rps-per-instance",
        type=float,
        default=60,
        help="Request throughput per model instance, R_ps (req/s).",
    )
    os_tp_group.add_argument(
        "--os-n-instances",
        type=int,
        default=1,
        help="Number of model instances, N_inst.",
    )
    os_tp_group.add_argument(
        "--os-utilization",
        type=float,
        default=0.9,
        help="GPU utilization fraction, U (0-1).",
    )

    # ---- Proprietary ----
    prop_group = parser.add_argument_group("proprietary — token pricing")
    prop_group.add_argument(
        "--prop-isl",
        type=int,
        default=1_000,
        help="Input sequence length per model call, ISL (tokens).",
    )
    prop_group.add_argument(
        "--prop-osl",
        type=int,
        default=500,
        help="Output sequence length per model call, OSL (tokens).",
    )
    prop_group.add_argument(
        "--prop-price-in-per-million",
        type=float,
        default=1.25,
        help="Vendor price per million input tokens, P_in (USD).",
    )
    prop_group.add_argument(
        "--prop-price-out-per-million",
        type=float,
        default=10.0,
        help="Vendor price per million output tokens, P_out (USD).",
    )
    prop_group.add_argument(
        "--prop-api-overhead",
        type=float,
        default=0.0,
        help="Fixed API overhead cost per call, C_API_Overhead (USD).",
    )

    # ---- Human ----
    human_group = parser.add_argument_group("human — labor costs")
    human_group.add_argument(
        "--human-wage-gross",
        type=float,
        default=12.0,
        help="Gross hourly wage of a worker, W_gross (USD/hr).",
    )
    human_group.add_argument(
        "--human-platform-fee-frac",
        type=float,
        default=0.20,
        help="Platform commission as a fraction, F_platform (e.g. 0.20).",
    )
    human_group.add_argument(
        "--human-time-per-task-seconds",
        type=float,
        default=600.0,
        help="Estimated task duration per worker, T_task (s).",
    )
    human_group.add_argument(
        "--human-qa-amortized-per-task",
        type=float,
        default=0.0,
        help="Amortized QA overhead cost per task, C_QA (USD).",
    )
    human_group.add_argument(
        "--human-n-humans",
        type=int,
        default=1,
        help="Number of human workers needed per task, N_Humans.",
    )

    main(parser.parse_args())
