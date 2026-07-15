from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from kubernetes import client, config

NAMESPACE = "llm-ollama"
ISVC_NAME = "ollama-llm"

CONTAINER_SPEC_TEMPLATE = {
    "name": "kserve-container",
    "image": "ollama/ollama:latest",
    "command": ["/bin/sh", "-c"],
    "args": [
        'ollama serve &\n'
        'until curl -sf http://127.0.0.1:11434/api/tags >/dev/null; do sleep 2; done\n'
        'ollama pull "$MODEL_REF"\n'
        'wait\n'
    ],
    "ports": [{"containerPort": 11434, "protocol": "TCP"}],
    "resources": {
        "requests": {"cpu": "2", "memory": "6Gi"},
        "limits": {"cpu": "4", "memory": "8Gi"},
    },
    "volumeMounts": [{"name": "ollama-models", "mountPath": "/root/.ollama"}],
}


def build_model_ref(hf_repository: str) -> str:
    hf_repository = hf_repository.strip()
    if ":" in hf_repository or hf_repository.startswith("hf.co/"):
        return hf_repository
    return f"hf.co/{hf_repository}"


def deploy_model(**context):
    hf_repository = context["dag_run"].conf.get("hf_repository")
    if not hf_repository:
        raise ValueError("hf_repository не передан в conf DAG run")

    model_ref = build_model_ref(hf_repository)

    container = dict(CONTAINER_SPEC_TEMPLATE)
    container["env"] = [
        {"name": "OLLAMA_HOST", "value": "0.0.0.0:11434"},
        {"name": "OLLAMA_KEEP_ALIVE", "value": "-1"},
        {"name": "MODEL_REF", "value": model_ref},
    ]

    config.load_incluster_config()
    custom = client.CustomObjectsApi()
    custom.patch_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=NAMESPACE,
        plural="inferenceservices",
        name=ISVC_NAME,
        body={"spec": {"predictor": {"containers": [container]}}},
    )

    core = client.CoreV1Api()
    pods = core.list_namespaced_pod(
        NAMESPACE,
        label_selector=f"serving.kserve.io/inferenceservice={ISVC_NAME}",
    )
    for pod in pods.items:
        core.delete_namespaced_pod(pod.metadata.name, NAMESPACE)


with DAG(
    dag_id="hf_model_downloader",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["llm", "kserve", "ollama"],
) as dag:
    PythonOperator(
        task_id="patch_and_restart_isvc",
        python_callable=deploy_model,
    )
