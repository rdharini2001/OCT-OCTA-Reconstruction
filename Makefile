.PHONY: run test dev build
run:
	cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
test:
	cd backend && python -m pytest -q
	cd frontend && npm run check
dev:
	cd frontend && npm run build
	cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
build:
	cd frontend && npm run build
