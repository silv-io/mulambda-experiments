import datetime
import json
import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from galileojp.k3s import K3SGateway
from influxdb_client.client.warnings import MissingPivotFunction
from util import json_transform

warnings.simplefilter("ignore", MissingPivotFunction)


def ecdf(gw: K3SGateway, exps: pd.DataFrame, metric: str):
    for idx, exp in exps.iterrows():
        ev = get_clear_events(gw, exp["EXP_ID"])
        meta = json.loads(exp["metadata"])
        sns.ecdfplot(data=ev, x=metric, label=meta["target"])
    plt.legend()
    plt.savefig(f"plots/ecdf-{metric}-{datetime.datetime.now()}.pdf")


def time_series(gw: K3SGateway, exp_ids: List[str]):
    for exp_id in exp_ids:
        ev = get_clear_events(gw, exp_id)
        sns.lineplot(data=ev, x="time", y="elapsed", label=exp_id)
    plt.savefig("test.png")


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


def filter(exps: pd.DataFrame, constraints: Dict
           ) -> pd.DataFrame:
    rows = []
    for idx, exp in exps.iterrows():
        j = json.loads(exp['metadata'])
        if matches(constraints, j):
            rows.append(exp)
    return pd.DataFrame(rows, columns=exps.columns)


if __name__ == '__main__':
    load_dotenv()
    gw = K3SGateway.from_env()
    exps = gw.experiments()

    test_exps = filter(exps, {"usecase": "scp", "amount": 10, "iterations": 5})
    print(test_exps)
    ecdf(gw, test_exps, "model_accuracy")
