from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="m1core-llm-portal")

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow-webserver.airflow.svc.cluster.local:8080")
AIRFLOW_DAG_ID = os.getenv("AIRFLOW_DAG_ID", "hf_model_downloader")
AIRFLOW_AUTH = (os.getenv("AIRFLOW_USER", "admin"), os.getenv("AIRFLOW_PASSWORD", "admin"))

class ModelDeployRequest(BaseModel):
    hf_url: str

@app.get("/health")
def health():
    return {"status": "up"}

@app.post("/api/v1/deploy")
def deploy_model(payload: ModelDeployRequest):
    hf_repo = payload.hf_url.replace("https://huggingface.co/", "")
    trigger_url = f"{AIRFLOW_URL}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"
    
    dag_config = {
        "conf": {
            "hf_repository": hf_repo,
            "initiated_by": "m1core-ui-portal"
        }
    }
    
    try:
        response = requests.post(trigger_url, json=dag_config, auth=AIRFLOW_AUTH, timeout=10)
        if response.status_code == 201:
            return {"status": "success", "message": f"Пайплайн для {hf_repo} запущен в Airflow!"}
        else:
            raise HTTPException(status_code=500, detail=f"Airflow Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Airflow unreachable: {str(e)}")
