# Safe deployment notes

Use `docker compose up --build` locally or run the FastAPI server on loopback. By design this repository does **not** expose public endpoints with authentication/authorization. Do not upload identifiable patient data on a shared server without an approved research data-handling protocol.

`OCTAVE_DATA_DIR` sets the directory for user-derived outputs. Never commit its contents. `OCTAVE_MAX_UPLOAD_MB` sets per-file upload limits (default 200). `OCTAVE_PREVIEW_SLICES` controls default preview sampling (128). `OCTAVE_SEGMENTER=legacy` selects the slower source-traced dynamic-programming boundary method.

The default inline background worker is in-process and job states do not persist across restarts. For hosted work, move to a durable task queue, authenticated user-scoped datasets, encrypted at-rest storage, virus/malware scans, per-tenant quotas, reverse-proxy request size limits and observability.
