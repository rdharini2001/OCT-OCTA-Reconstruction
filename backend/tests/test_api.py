import os
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

def test_api_empty_library_without_demo(monkeypatch,tmp_path):
    import app.main as main
    monkeypatch.setattr(main,'DATA_ROOT',tmp_path)
    with TestClient(app) as client:
        assert client.get('/api/health').json()['version']=='0.2.0'
        assert client.get('/api/datasets').json()==[]
        assert client.get('/api/datasets/synthetic-demo').status_code==404
        assert client.get('/').status_code==200
        assert client.get('/api/datasets/../../etc/passwd').status_code in (400,404)

def test_invalid_pipeline_mode_rejected(tmp_path):
    with TestClient(app) as client:
        r=client.post('/api/datasets',data={'mode':'fictional'})
        assert r.status_code==422
