import uuid

import pandas as pd
from kubernetes import client, config, watch

NODE_HOSTS = [f"s30-worker-zone-c-{node_id}" for node_id in range(4)] + ["s20-controller-zone-c-0"]

def short_uid():
    return str(uuid.uuid4().hex[:8])


def json_transform(data: pd.DataFrame, column: str, prefix: str = "") -> pd.DataFrame:
    """
    takes the json data in the value column of the dataframe and transforms it into extra columns
    for each key in the json
    :param data:
    :param column:
    :param prefix:
    :return:
    """
    key_values = data[column].apply(
        lambda x: list(x.keys()) if x is not None else []).explode().unique()

    for key in key_values:
        data[f"{prefix}{key}"] = data[column].apply(
            lambda x: x.get(key, None) if x is not None else None)

    data = data.drop(columns=[column])
    return data


def create_experiment_job(exp_id, client_id, usecase, amount, size,
                          iterations):
    config.load_kube_config(
        config_file="/home/silv/exp-cluster/config.yaml")

    api_instance = client.BatchV1Api()

    exp_name = f"{client_id}-{usecase}-{amount}-{size}-{iterations}"
    if len(exp_name) > 63:
        exp_name = exp_name[:63]

    # Define the Job object
    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=f"exp-{exp_id}-{exp_name}",
                                     namespace="mulambda"),
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="experiment-job",
                            image="agihi/mulambda:latest",
                            command=["make", "run-experiment"],
                            image_pull_policy="Always",
                            env=[
                                client.V1EnvVar(name="MULAMBDA_EXPERIMENT__NAME",
                                                value=exp_name),
                                client.V1EnvVar(name="MULAMBDA_EXPERIMENT__ID",
                                                value=exp_id),
                                client.V1EnvVar(name="MULAMBDA_EXPERIMENT__TARGET",
                                                value=client_id),
                                client.V1EnvVar(name="MULAMBDA_EXPERIMENT__USECASE",
                                                value=usecase),
                                client.V1EnvVar(name="MULAMBDA_EXPERIMENT__AMOUNT",
                                                value=str(amount)),
                                client.V1EnvVar(name="MULAMBDA_EXPERIMENT__SIZE",
                                                value=str(size)),
                                client.V1EnvVar(name="MULAMBDA_EXPERIMENT__ITERATIONS",
                                                value=str(iterations)),
                            ],
                        )
                    ],
                )
            )
        )
    )

    # Create the Job
    api_response = api_instance.create_namespaced_job(body=job, namespace="mulambda")
    print("Job created. Status='%s'" % str(api_response.status))


def wait_for_job_completion(job_name: str):
    config.load_kube_config()  # Use this if running the code outside the cluster

    api_instance = client.BatchV1Api()

    # Watch the Job events
    w = watch.Watch()
    for event in w.stream(api_instance.list_namespaced_job, namespace="mulambda",
                          field_selector=f"metadata.name={job_name}"):
        job = event['object']
        if job.status.active is not None and job.status.active == 0:
            # The Job has completed
            w.stop()
            print("Job completed.")
            return
        elif job.status.failed is not None and job.status.failed > 0:
            # The Job has failed
            w.stop()
            print("Job failed.")
            return
