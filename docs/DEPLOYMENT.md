# Deploying OCTAVE

**Recommended environment:** local machine, institutional private workstation or VPN-protected host. The API has no authentication, authorization or audit log and is **not safe as a public upload endpoint**.

1. `cd backend && python -m pip install -r requirements.txt`.
2. `uvicorn app.main:app --host 127.0.0.1 --port 8000`.
3. Open `http://127.0.0.1:8000` and import a study.

A ready frontend is committed under `frontend/dist/`. To update it: `cd frontend && npm run check`. No npm install or React build is needed. Docker Compose: `docker compose up --build`, bound to `127.0.0.1:8000` by default.

Set `OCTAVE_DATA_DIR` to a private writable disk location with sufficient capacity. Full OCT/flow processing can consume substantial RAM. The server currently uses a single in-process worker queue and job status is lost if the server restarts; implement durable workers and cleanup/deletion policies before production use. Configure de-identification, TLS, authentication, upload scanning, retention and disk quotas before collaboration over a network. Back up original inputs separately; the app deletes the uploaded archives after conversion.
