.PHONY: run test dev build
run:
	cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
test:
	cd backend && python -m pytest -q
dev:
	cd frontend && npm run dev
build:
	cd frontend && npm run build
