# Contributing

The UI source is `frontend/src/app.js` and `frontend/src/styles.css`; build with `cd frontend && npm run check`, without installing dependencies. The backend is in `backend/app`, with the original supplied v5 pipeline in `backend/octa_pipeline`. Run `cd backend && python -m pytest -q` before submitting changes. Never commit patient data, model weights without redistribution permission, unverified mm measurements or screenshots derived from identifiable scans. Document calibration, QC and the source of each algorithm change.
