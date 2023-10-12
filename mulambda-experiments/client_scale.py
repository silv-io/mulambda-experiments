from kubernetes import client, config


def create_deployment_and_service(client_id, target_selector):
    # Load the default Kubernetes configuration from your kubeconfig file
    config.load_kube_config()

    # Create a Kubernetes API instance for Deployments and Services
    apps_v1 = client.AppsV1Api()
    v1 = client.CoreV1Api()

    # Create the Deployment
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=client_id, namespace="mulambda"),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(match_labels={"app": client_id}),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": client_id}),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name="mulambda",
                            image="agihi/mulambda:latest",
                            command=["make", "run-client"],
                            image_pull_policy="Always",
                            ports=[client.V1ContainerPort(container_port=80)],
                            env=[
                                client.V1EnvVar(name="MULAMBDA_CLIENT__ID",
                                                value=client_id),
                                client.V1EnvVar(name="MULAMBDA_NETWORK__SELECTOR",
                                                value=target_selector),
                            ],
                        )
                    ],
                    node_selector={
                        "ether.edgerun.io/zone": "zone-c",
                        "node-role.kubernetes.io/controller": "true",
                    },
                ),
            ),
        ),
    )

    # Create the Deployment
    apps_v1.create_namespaced_deployment(
        body=deployment, namespace="mulambda"
    )

    # Create the Service
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=client_id, namespace="mulambda"),
        spec=client.V1ServiceSpec(
            selector={"app": client_id},
            ports=[client.V1ServicePort(protocol="TCP", port=80, target_port=80)],
            type="ClusterIP",
        ),
    )

    # Create the Service
    v1.create_namespaced_service(body=service, namespace="mulambda")





if __name__ == '__main__':
    target_selectors = ["mulambda-selector", "plain-net-latency-selector",
                        "random-selector", "round-robin-selector"]

    # create experiment job

    for target_selector in target_selectors:
        for i in range(1, num_clients_per_target + 1):
            client_id = f"{target_selector}-client-{i}"
            create_deployment_and_service(client_id, target_selector)


