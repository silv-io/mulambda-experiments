import json
import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from galileojp.k3s import K3SGateway
from influxdb_client.client.warnings import MissingPivotFunction
from util import json_transform, mkdir, short_uid

warnings.simplefilter("ignore", MissingPivotFunction)

usecases = {
    "env": "Environment",
    "scp": "Smart-City Pedestrian Safety",
    "mda": "Medical Diagnosis Assistance",
    "psa": "Public Sentiment Analysis",
}

clients = {
    "mulambda-client": "Weighted",
    "plain-net-latency-client": "Plain Network Latency",
    "random-client": "Random",
    "round-robin-client": "Round Robin",
}

colors = {
    "mulambda-client": "#9748FF",
    "plain-net-latency-client": "#FFBF00",
    "random-client": "#FF6BAB",
    "round-robin-client": "#42C268",
}

util_colors = {
    "mulambda-client": ["#9748FF", "#ab6cff", "#c59aff", "#d5b5ff"],
    "plain-net-latency-client": ["#FFBF00", "#ffcb32", "#ffdb72", "#ffe599"],
    "random-client": ["#FF6BAB", "#ff88bb", "#feadd0", "#ffbcd9"],
    "round-robin-client": ["#42C268", "#67ce86", "#8ddaa4", "#a9e3bb"],
}

metrics = {
    "elapsed": "RTT",
    "model_accuracy": "Accuracy",
}


def _build_title(plot_type: str, metric: str, usecase: str, schedule: str):
    return (f"{plot_type} of {metrics[metric]} for {usecases[usecase]}\n"
            f"in {schedule.capitalize()} schedule")


def ecdf(gw: K3SGateway, exps: pd.DataFrame, metric: str, constraints: Dict,
         schedule: str = "Unknown",
         base_dir: str = "plots",
         parent_ax=None,
         format: str = "pdf"):
    if parent_ax is None:
        ax = plt.gca()
    else:
        ax = parent_ax
    exps["Target"] = exps["metadata"].apply(lambda x: json.loads(x)["target"])
    groups = exps.groupby("Target")["EXP_ID"].apply(list).reset_index()
    for idx, group in groups.iterrows():
        print(f"Generating ECDF for {group['Target']} with {group['EXP_ID']}")
        ev = pd.concat([get_clear_events(gw, exp_id) for exp_id in group["EXP_ID"]])
        sns.ecdfplot(data=ev, x=metric,
                     label=clients[group["Target"]], color=colors[group["Target"]],
                     ax=ax)
    ax.set_title(_build_title("ECDF", metric, constraints["usecase"], schedule))
    if metric == "elapsed":
        ax.set_xlabel("RTT [s]")
    if metric == "model_accuracy":
        ax.set_xlabel("Accuracy [%]")
    ax.legend()
    if parent_ax is None:
        plt.savefig(
            f"{base_dir}/ecdf-{metric}-{constraints['usecase']}-{schedule}.{format}")
        plt.clf()


def boxplot(gw: K3SGateway, exps: pd.DataFrame, metric: str, constraints: Dict,
            schedule: str = "Unknown",
            base_dir: str = "plots",
            parent_ax=None):
    if parent_ax is None:
        ax = plt.gca()
    else:
        ax = parent_ax
    exps["Target"] = exps["metadata"].apply(lambda x: json.loads(x)["target"])
    groups = exps.groupby("Target")["EXP_ID"].apply(list).reset_index()
    all = []
    for idx, group in groups.iterrows():
        print(f"Generating Boxplot for {group['Target']} with {group['EXP_ID']}")
        ev = pd.concat([get_clear_events(gw, exp_id) for exp_id in group["EXP_ID"]])
        ev["Target"] = clients[group["Target"]]
        all.append(ev)
    data = pd.concat(all, ignore_index=True)
    sns.boxplot(data=data, x="Target", y=metric, ax=ax, palette=colors.values())
    ax.set_xlabel(None)
    ax.set_title(_build_title("Boxplot", metric, constraints["usecase"], schedule))
    if metric == "elapsed":
        ax.set_ylabel("RTT [s]")
    if metric == "model_accuracy":
        ax.set_ylabel("Accuracy [%]")
    if parent_ax is None:
        plt.savefig(
            f"{base_dir}/boxplot-{metric}-{constraints['usecase']}-{schedule}.pdf")
        plt.clf()


def time_series(gw: K3SGateway, exps: pd.DataFrame, metric: str, constraints: Dict,
                schedule: str = "Unknown",
                rolling: bool = True, base_dir: str = "plots", format: str = "pdf"):
    plt.tight_layout()
    plt.figure(figsize=(10, 5))
    for idx, exp in exps.iterrows():
        ev = get_clear_events(gw, exp["EXP_ID"])
        meta = json.loads(exp["metadata"])
        if rolling:
            window_size = int(len(ev[metric]) / 60)
            ev["rolling"] = ev[metric].rolling(window_size).mean()
            ev['deviation'] = ev[metric].rolling(window_size).std()
            ev['upper'] = ev['rolling'] + 1.645 * ev['deviation']
            ev['lower'] = ev['rolling'] - 1.645 * ev['deviation']
            ev['lower'] = ev['lower'].clip(lower=0)
            sns.lineplot(data=ev, x="time", y="rolling", label=clients[meta["target"]],
                         color=colors[meta["target"]], linewidth=0.8)
            plt.fill_between(ev["time"], ev["lower"], ev["upper"], alpha=0.2,
                             color=colors[meta["target"]])
        else:
            sns.lineplot(data=ev, x="time", y=metric, label=clients[meta["target"]],
                         linewidth=0.8, color=colors[meta["target"]])
    plt.title(_build_title("Time Series", metric, constraints["usecase"], schedule))
    plt.legend()
    plt.xlabel("Time since start [s]")
    if metric == "elapsed":
        plt.ylabel("RTT [s]")
    if metric == "model_accuracy":
        plt.ylabel("Accuracy [%]")
    plt.grid(True)
    plt.savefig(
        f"{base_dir}/ts-{metric}-{constraints['usecase']}-{schedule}-{short_uid()}.{format}")
    plt.clf()


def get_clear_events(gw: K3SGateway, exp_id: str) -> pd.DataFrame:
    ev = gw.events(exp_id)
    ev["value"] = ev["value"].apply(json.loads)
    transformed = json_transform(ev, "value")
    transformed = json_transform(transformed, "model_traits", "model_")
    transformed["time"] = (transformed.index - min(transformed.index)).total_seconds()
    return transformed


def matches(constraints, item):
    if type(item) is list:
        return set(constraints) == set(item)
    elif type(item) is dict:
        if len(item) == 0:
            return False
        for k, v in constraints.items():
            j = item.get(k, None)
            if j is None:
                return False
            if matches(v, j) is False:
                return False
    elif item != constraints:
        return False
    return True


def filter(exps: pd.DataFrame, constraints: Dict,
           id_prefices: str | List[str]) -> pd.DataFrame:
    if type(id_prefices) is str:
        id_prefices = [id_prefices]
    rows = []
    for idx, exp in exps.iterrows():
        j = json.loads(exp['metadata'])
        if matches(constraints, j) and any(
                j["exp_id"].startswith(id_prefix) for id_prefix in id_prefices):
            rows.append(exp)
    return pd.DataFrame(rows, columns=exps.columns)


def extract_schedule(name: str) -> str:
    if "-log-" in name or "-logical-" in name:
        return "logical"
    if "-arb-" in name:
        return "arbitrary"
    if "-par-" in name:
        return "parity"
    if "-loc-" in name:
        return "hyperlocal"


def generate_base_plots(gw: K3SGateway, exps: pd.DataFrame, constraints: Dict,
                        prefices: List[str], base_dir: str = "plots",
                        format: str = "pdf"):
    logical_exps = filter(exps[exps["SCHEDULE"] == "logical"], constraints,
                          id_prefices=prefices)
    arbitrary_exps = filter(exps[exps["SCHEDULE"] == "arbitrary"], constraints,
                            id_prefices=prefices)
    if len(logical_exps) < 4 or len(arbitrary_exps) < 4:
        return
    print(
        f"Generating base plots for {constraints['usecase']} "
        f"with logical: {logical_exps.NAME.tolist()} "
        f"and arbitrary: {arbitrary_exps.NAME.tolist()}")
    for metric in ["model_accuracy", "elapsed"]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        ecdf(gw, logical_exps, metric, constraints, schedule="logical",
             parent_ax=axes[0][0])
        boxplot(gw, logical_exps, metric, constraints, schedule="logical",
                parent_ax=axes[0][1])
        ecdf(gw, arbitrary_exps, metric, constraints, schedule="arbitrary",
             parent_ax=axes[1][0])
        boxplot(gw, arbitrary_exps, metric, constraints, schedule="arbitrary",
                parent_ax=axes[1][1])
        plt.tight_layout()
        plt.savefig(
            f"{base_dir}/ecdfbox-{metric}-{constraints['usecase']}-base.{format}")
        plt.clf()


def generate_edge_plots(gw: K3SGateway, exps: pd.DataFrame, constraints: Dict,
                        prefices: List[str], base_dir: str = "plots",
                        format: str = "pdf"):
    hyperlocal_exps = filter(exps[exps["SCHEDULE"] == "hyperlocal"], constraints,
                             id_prefices=prefices)
    parity_exps = filter(exps[exps["SCHEDULE"] == "parity"], constraints,
                         id_prefices=prefices)
    if len(hyperlocal_exps) < 4 or len(parity_exps) < 4:
        return
    print(
        f"Generating edge plots for {constraints['usecase']} "
        f"with hyperlocal: {hyperlocal_exps.NAME.tolist()} "
        f"and parity: {parity_exps.NAME.tolist()}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ecdf(gw, hyperlocal_exps, "elapsed", constraints, schedule="hyperlocal",
         parent_ax=axes[0])
    ecdf(gw, hyperlocal_exps, "model_accuracy", constraints, schedule="hyperlocal",
         parent_ax=axes[1])
    plt.tight_layout()
    plt.savefig(f"{base_dir}/ecdf-{constraints['usecase']}-hyperlocal.{format}")
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ecdf(gw, parity_exps, "elapsed", constraints, schedule="parity",
         parent_ax=axes[0])
    ecdf(gw, parity_exps, "model_accuracy", constraints, schedule="parity",
         parent_ax=axes[1])
    plt.tight_layout()
    plt.savefig(f"{base_dir}/ecdf-{constraints['usecase']}-parity.{format}")
    plt.clf()


UTIL_INTRO_PREFICE = "20231019"


def generate_utilization(gw: K3SGateway, exps: pd.DataFrame, constraints: Dict,
                         prefices: List[str], base_dir: str = "plots",
                         format: str = "pdf"):
    utilization_prefices = [p for p in prefices if p >= UTIL_INTRO_PREFICE]
    for prefix in utilization_prefices:
        test_exps = filter(exps[exps["SCHEDULE"] == "logical"], constraints,
                           id_prefices=prefix)
        if len(test_exps) < 4:
            continue
        print(
            f"Generating utilization for {constraints['usecase']} "
            f"with {test_exps.NAME.tolist()}"
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        fig.suptitle(
            f"Resource Utilization for {usecases[constraints['usecase']]} per selection algorithm")
        test_exps["Target"] = test_exps["metadata"].apply(
            lambda x: json.loads(x)["target"])
        groups = test_exps.groupby("Target")["EXP_ID"].apply(list).reset_index()
        for idx, group in groups.iterrows():
            ax = axes[int(idx / 2)][idx % 2]
            print(
                f"Generating Utilization for {group['Target']} with {group['EXP_ID']}")
            if len(group["EXP_ID"]) > 1:
                print("ERROR: More than one experiment per target")
            ev = pd.concat([get_clear_events(gw, exp_id) for exp_id in group["EXP_ID"]])
            ev = get_clear_events(gw, group["EXP_ID"][0])
            reqs = ev[ev["type"] == "request"].copy()
            reqs["targeted_node"] = reqs["targeted_node"].astype("int")
            reqs["start_time"] = reqs.index - pd.to_timedelta(reqs["elapsed"], unit="s")

            time_intervals = pd.date_range(
                start=min(reqs["start_time"]),
                end=max(reqs.index),
                freq="100L"
            )
            ev_counts = []
            nodes = reqs["targeted_node"].unique()

            for interval in time_intervals:
                counts_per_node = [0, 0, 0, 0]
                for node in nodes:
                    count = ((reqs['start_time'] <= interval) & (
                            reqs.index >= interval) & (
                                     reqs["targeted_node"] == node)).sum()
                    counts_per_node[node] = count
                ev_counts.append(counts_per_node)

            counts_df = pd.DataFrame(ev_counts)

            counts_df.columns = ["Node 1", "Node 2", "Node 3", "Node 4"]
            counts_df.index = counts_df.index * 0.1
            counts_df.plot.area(ax=ax, stacked=True,
                                color=util_colors[group["Target"]], linewidth=0)
            ax.set_title(f"{clients[group['Target']]}")
            ax.set_xlabel("Time since start [s]")
            ax.set_ylabel("Number of Requests")
        plt.tight_layout()
        plt.savefig(f"{base_dir}/util-{constraints['usecase']}-{prefix}.{format}")
        plt.clf()


def generate_psa_shift():
    accuracy_weights = [0.5] * 20 + [0.8] * 20 + [1.0] * 20 + [0.2] * 20 + [0] * 20
    latency_weights = [0.5] * 20 + [0.2] * 20 + [0] * 20 + [0.8] * 20 + [1.0] * 20
    df = pd.DataFrame({"Accuracy": accuracy_weights, "Latency": latency_weights})
    plt.figure(figsize=(10, 5))
    plt.tight_layout()
    colors = {
        "Accuracy": "#069e2d",
        "Latency": "#4357ad"
    }
    sns.set_style("whitegrid")
    for column in df.columns:
        sns.lineplot(data=df, x=df.index, y=column, dashes=False, color=colors[column],
                     label=column)
    plt.legend()
    plt.xlabel("Percent of passed requests in experiment")
    plt.ylabel("Absolute Weights [0..1]")
    plt.title("Weight Distribution Shift for ENV")
    plt.savefig("plots/debug/env-shift.pdf")
    plt.clf()


def generate_days(gw: K3SGateway, prefices: List[str] | str):
    if type(prefices) is str:
        prefices = [prefices]
    base_dir = f"plots/{'-'.join(prefices)}"
    mkdir(base_dir)

    exps = gw.experiments()
    exps["SCHEDULE"] = exps.NAME.apply(extract_schedule)

    for usecase in ["scp", "mda", "psa", "env"]:
        constraints = {"usecase": usecase, "iterations": 5}
        # generate_base_plots(gw, exps, constraints, prefices, base_dir=base_dir,
        #                     format="pdf")
        # generate_edge_plots(gw, exps, constraints, prefices, base_dir=base_dir,
        #                     format="pdf")
        generate_utilization(gw, exps, constraints, prefices, base_dir=base_dir,
                             format="pdf")
        # for schedule in ["logical", "arbitrary", "parity", "hyperlocal"]:
        #     test_exps = filter(exps[exps["SCHEDULE"] == schedule], constraints,
        #                        id_prefices=prefices)
        #     if len(test_exps) < 4:
        #         print(f"Skipping {schedule} for {usecase}")
        #         continue
        #     print(
        #         f"Generating {schedule} for {usecase} with: {test_exps.NAME.tolist()}")
        #     for metric in ["model_accuracy", "elapsed"]:
        #         ecdf(gw, test_exps, metric, constraints, schedule=schedule,
        #              base_dir=base_dir, format="pdf")
        # for prefix in prefices:
        #     test_exps = filter(exps[exps["SCHEDULE"] == "logical"], constraints,
        #                        id_prefices=prefix)
        #     if len(test_exps) < 4:
        #         continue
        #     for metric in ["model_accuracy", "elapsed"]:
        #         time_series(gw, test_exps, metric, constraints,
        #                     schedule="logical",
        #                     base_dir=base_dir, format="pdf")


def extract_scale_type(name: str) -> str:
    if "-it-scale-" in name:
        return "Concurrency"
    if "-size-scale-" in name:
        return "Data Size"
    else:
        return "None"


def generate_scale(gw: K3SGateway, prefices: List[str] | str):
    if type(prefices) is str:
        prefices = [prefices]
    base_dir = f"plots/scale-{'-'.join(prefices)}"
    mkdir(base_dir)

    exps = gw.experiments()
    exps["SCALE_TYPE"] = exps.NAME.apply(extract_scale_type)
    index = [1, 3, 5, 8, 13, 21, 34, 55, 89]
    columns = ["Scale", "Selector", "Metric", "Average", "Median", "StD", "Upper",
               "Lower"]

    aggregates = {}
    for scale_type in ["Concurrency", "Data Size"]:
        aggregates[scale_type] = pd.DataFrame(columns=columns)
        for scale in index:
            if scale_type == "Concurrency":
                constraints = {"usecase": "psa", "iterations": scale}
            elif scale_type == "Data Size":
                constraints = {"usecase": "psa", "size": scale}
            else:
                break
            test_exps = filter(exps[exps["SCALE_TYPE"] == scale_type], constraints,
                               id_prefices=prefices)
            print(test_exps.NAME.tolist())
            for exp in test_exps.iterrows():
                exp_id = exp[1]["EXP_ID"]
                target = json.loads(exp[1]["metadata"])["target"]
                ev = get_clear_events(gw, exp_id)
                for metric in ["elapsed", "model_accuracy"]:
                    avg = ev[metric].mean()
                    median = ev[metric].median()
                    std = ev[metric].std()
                    if np.isnan(std):
                        std = 0
                    aggregates[scale_type] = aggregates[scale_type].append({
                        "Scale": scale,
                        "Selector": target,
                        "Metric": metric,
                        "Average": avg,
                        "Median": median,
                        "StD": std,
                        "Upper": avg + 1.645 * std,
                        "Lower": max(avg - 1.645 * std, 0)
                    }, ignore_index=True)
        aggregates[scale_type]["Scale"] = pd.to_numeric(aggregates[scale_type]["Scale"])
        aggregates[scale_type].reset_index(drop=True, inplace=True)

    for scale_type in ["Concurrency", "Data Size"]:
        scale_df = aggregates[scale_type]
        for metric in ["elapsed", "model_accuracy"]:
            plt.tight_layout()
            plt.figure(figsize=(10, 5))
            metric_df = scale_df[scale_df["Metric"] == metric]
            for selector in ["mulambda-client", "plain-net-latency-client",
                             "random-client", "round-robin-client"]:
                selector_df = metric_df[metric_df["Selector"] == selector]
                sns.lineplot(data=selector_df, x="Scale",
                             y="Average", label=clients[selector],
                             color=colors[selector])
                plt.fill_between(selector_df["Scale"], selector_df["Lower"],
                                 selector_df["Upper"], alpha=0.1,
                                 color=colors[selector])
            if metric == "elapsed":
                plt.ylabel("Average RTT [s]")
            if metric == "model_accuracy":
                plt.ylabel("Average Accuracy [%]")
            if scale_type == "Concurrency":
                plt.xlabel("Number of concurrent requests")
            if scale_type == "Data Size":
                plt.xlabel("Abstract size of data")
            plt.title(f"Average {metrics[metric]} when scaling {scale_type}")
            plt.savefig(f"{base_dir}/scale-{scale_type}-{metric}.pdf")
            plt.clf()


def debug(gw: K3SGateway):
    mkdir("plots/debug")
    constraints = {"usecase": "scp", "iterations": 5}
    exps = gw.experiments()
    exps["SCHEDULE"] = exps.NAME.apply(extract_schedule)
    test_exps = filter(exps[exps["SCHEDULE"] == "logical"], constraints, ["20230928"])
    boxplot(gw, test_exps, "elapsed", constraints, schedule="logical",
            base_dir="plots/debug")


if __name__ == '__main__':
    load_dotenv()
    gw = K3SGateway.from_env()
    pd.set_option('display.max_colwidth', None)
    # debug(gw)
    generate_days(gw, ["20231005", "20230928", "20230922", "20231012", "20231019"])
    # generate_psa_shift()
    # generate_scale(gw, ["20231025"])
