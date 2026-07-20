from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import os
import uvicorn

app = FastAPI(title="m1core-llm-portal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow-api-server.airflow.svc.cluster.local:8080")
AIRFLOW_DAG_ID = os.getenv("AIRFLOW_DAG_ID", "hf_model_downloader")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")


class ModelDeployRequest(BaseModel):
    hf_url: str


@app.get("/health")
def health():
    return {"status": "up"}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>M1 Core LLM Portal</title>
        <style>
            body { font-family: sans-serif; max-width: 600px; margin: 60px auto; padding: 0 20px; }
            input { width: 100%; padding: 10px; font-size: 16px; box-sizing: border-box; }
            button { margin-top: 12px; padding: 10px 20px; font-size: 16px; cursor: pointer; }
            #result { margin-top: 20px; padding: 12px; border-radius: 4px; white-space: pre-wrap; }
            .success { background: #e6ffed; border: 1px solid #34c759; }
            .error { background: #ffe6e6; border: 1px solid #ff3b30; }
        </style>
    </head>
    <body>
        <h2>M1 Core LLM Portal</h2>
        <p>Вставьте HuggingFace-путь модели (например huihui_ai/qwen3-abliterated) или Ollama-тег вида org/model:tag.</p>
        <input type="text" id="hfUrl" placeholder="huihui_ai/qwen3-abliterated:1.7b">
        <button onclick="deploy()">Загрузить модель</button>
        <div id="result"></div>

        <script>
            async function deploy() {
                const url = document.getElementById('hfUrl').value;
                const resultDiv = document.getElementById('result');
                resultDiv.className = '';
                resultDiv.textContent = 'Отправка запроса...';

                try {
                    const response = await fetch('/api/v1/deploy', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ hf_url: url })
                    });
                    const data = await response.json();

                    if (response.ok) {
                        resultDiv.className = 'success';
                        resultDiv.textContent = data.message;
                    } else {
                        resultDiv.className = 'error';
                        resultDiv.textContent = 'Ошибка: ' + (data.detail || 'неизвестная ошибка');
                    }
                } catch (e) {
                    resultDiv.className = 'error';
                    resultDiv.textContent = 'Не удалось связаться с сервером: ' + e.message;
                }
            }
        </script>
    </body>
    </html>
    """


def get_airflow_token() -> str:
    resp = requests.post(
        f"{AIRFLOW_URL}/auth/token",
        json={"username": AIRFLOW_USER, "password": AIRFLOW_PASSWORD},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


@app.post("/api/v1/deploy")
def deploy_model(payload: ModelDeployRequest):
    hf_repo = payload.hf_url.replace("https://huggingface.co/", "").strip("/")
    trigger_url = f"{AIRFLOW_URL}/api/v2/dags/{AIRFLOW_DAG_ID}/dagRuns"

    dag_config = {
        "conf": {
            "hf_repository": hf_repo,
            "initiated_by": "m1core-ui-portal",
        },
        "logical_date": None,
    }

    try:
        token = get_airflow_token()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Airflow auth failed: {str(e)}")

    try:
        response = requests.post(
            trigger_url,
            json=dag_config,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if response.status_code in (200, 201):
            return {"status": "success", "message": f"Пайплайн для {hf_repo} запущен в Airflow!"}
        else:
            raise HTTPException(status_code=500, detail=f"Airflow Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Airflow unreachable: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
