# Publish OCTAVE on GitHub

This is a **GitHub-ready ZIP**, not an automatically published GitHub repository.

1. Extract the package to a clean folder and inspect the contents.
2. Create a new private repository in GitHub.
3. In the extracted `octave-research-studio` folder:

```bash
git init
git add .
git commit -m "OCTAVE v0.2: v5/v6 pipeline and imaging workstation"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/octave-research-studio.git
git push -u origin main
```

The CI workflow runs `python -m pytest -q` under Python 3.11 and checks the **single-source dependency-free frontend** with `npm run check` under Node 22. Rebuilt `frontend/dist/` must match committed outputs. If CI fails, read the actual workflow logs; a red badge alone does not identify the cause.

Before making public: verify the licenses of the supplied `backend/octa_pipeline/` and `research/reference-scripts/`, scrub all patient identifiers, and keep `runtime-data/` / input archives private. Never publish patient OCT volumes without explicit permission.
