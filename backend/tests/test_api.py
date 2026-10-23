import os
from fastapi.testclient import TestClient
from app.main import app


def test_api_health_and_demo(tmp_path):
    with TestClient(app) as c:
        assert c.get('/api/health').status_code==200
        datasets=c.get('/api/datasets').json()
        assert any(d['id']=='synthetic-demo' for d in datasets)
        r=c.get('/api/datasets/synthetic-demo/bscan/50')
        assert r.status_code==200 and r.headers['content-type']=='image/png'
        r=c.get('/api/datasets/synthetic-demo/enface/svc')
        assert r.status_code==200
        r=c.get('/api/datasets/synthetic-demo/analyze/svc')
        assert r.status_code==200 and 'faz_detected' in r.json()
        assert c.get('/api/datasets/synthetic-demo/surface?step=8').status_code==200
        assert c.get('/api/datasets/synthetic-demo/export').status_code==200
        assert c.get('/api/datasets/synthetic-demo/bscan/999').status_code==404
        assert c.get('/api/datasets/../../etc/passwd').status_code in (400,404)
