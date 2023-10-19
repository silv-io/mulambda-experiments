import json
import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
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
    plt.figure(figsize=(8, 5))
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
                        prefices: List[str], base_dir: str = "plots", format: str = "pdf"):
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
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ecdf(gw, logical_exps, metric, constraints, schedule="logical",
             parent_ax=axes[0][0])
        boxplot(gw, logical_exps, metric, constraints, schedule="logical",
                parent_ax=axes[0][1])
        ecdf(gw, arbitrary_exps, metric, constraints, schedule="arbitrary",
             parent_ax=axes[1][0])
        boxplot(gw, arbitrary_exps, metric, constraints, schedule="arbitrary",
                parent_ax=axes[1][1])
        plt.tight_layout()
        plt.savefig(f"{base_dir}/ecdfbox-{metric}-{constraints['usecase']}-base.{format}")
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
        generate_base_plots(gw, exps, constraints, prefices, base_dir=base_dir, format="png")
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
        #              base_dir=base_dir, format="png")
        for prefix in prefices:
            test_exps = filter(exps[exps["SCHEDULE"] == "logical"], constraints,
                               id_prefices=prefix)
            if len(test_exps) < 4:
                continue
            for metric in ["model_accuracy", "elapsed"]:
                time_series(gw, test_exps, metric, constraints,
                            schedule="logical",
                            base_dir=base_dir, format="png")


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
    generate_days(gw, ["20231005", "20230928", "20230922", "20231012"])
