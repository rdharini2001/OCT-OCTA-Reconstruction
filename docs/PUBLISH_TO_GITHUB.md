# Publishing OCTAVE to GitHub

This ZIP is a complete, **GitHub-ready source repository**. It is not automatically published to a GitHub account. Review the repository and data-protection notes before choosing public or private visibility.

## Recommended: start private

1. On GitHub, create an empty private repository named `octave-research-studio` (do not add a README or license there).
2. Extract this archive and run the following from the extracted `octave-research-studio` folder:

```bash
git init
git add .
git commit -m "Initial OCTAVE research studio"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/octave-research-studio.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your account or organization. You can use GitHub CLI after sign-in instead: `gh repo create octave-research-studio --private --source . --remote origin --push`.

## Before making it public

- Confirm that you have rights to redistribute `backend/octa_pipeline/`, which came from uploaded user-provided code without a visible license. The project MIT license **does not cover** this imported package by default. You can keep the repository private pending clearance.
- Verify no identifiable images, patient names, acquisition IDs, tokens, or `.env` secrets are present. The release ZIP intentionally excludes raw scans and the local `runtime-data` directory.
- Confirm scientific method descriptions are accurate for your scanner and flow input. The reconstruction and segmentation methods are exploratory and not clinically validated.
- Run the Python backend tests and the React build in GitHub Actions. The standalone `frontend/dist/` is included for immediate Python-only use.

## Local operations

The API provides no login system and holds studies on the local server. Keep it bound to localhost until authentication, authorization, encryption, study deletion, and governance controls are added.
