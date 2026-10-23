# Source provenance and license notices

- The OCTAVE frontend, FastAPI glue, synthetic phantom, adapters, tests, scripts, documentation, and new experimental segmentation helper were created for this project and are made available under `LICENSE` (MIT).
- `backend/octa_pipeline/` was taken from the user's uploaded `octa_pipeline (2).zip`. It contains user-supplied reconstruction research code. Its uploaded copy did not include an explicit license. It is included to preserve functional compatibility with the user's work, **not** because it was independently proven eligible for public redistribution. Confirm permission before pushing this repository publicly; you may need to keep it private.
- The uploaded FAZ script inspired seeded, enclosed non-vessel component detection. The application implementation of that approach is in `backend/app/analysis.py`; the original FAZ repository is not bundled.
- `Choroidalyzer` and `OCTolyzer` are separate projects, not included in the app. There are no bundled trained weights or externally sourced patient images. The term “Choroidalyzer” in documentation means potential/optional integration only, not a claim that its segmenter is active.
- The included synthetic data generator and preview screenshots are simulated research visualizations and **not** evidence of OCTA reconstruction accuracy.

This is research software. Neither the packaging license nor visual design constitutes clinical clearance, measurement accuracy certification, or permission to redistribute restricted clinical data.
