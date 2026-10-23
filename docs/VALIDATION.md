# OCTAVE — Scientific and technical validation checklist

This project is a runnable **research scaffold**, not an FDA/CE-cleared medical device and not a clinically validated OCTA image generator.

## Before using reported measurements in a paper

- [ ] Record vendor, model, serial/firmware, scan protocol, slab definitions, repeated B-scan protocol, physical calibration and eye laterality.
- [ ] Confirm B-scan order, direction, registration and image orientation; repeated OCT B-scans are not equivalent to a registered flow volume.
- [ ] Validate ILM, IPL/INL, OPL/ONL and RPE boundaries with masked expert annotations across normal eyes and pathology.
- [ ] Replace geometric IPL/INL, OPL/ONL and choroidal proxies with modality-trained boundaries and document model-domain limits.
- [ ] Quantify FAZ segmentation Dice/IoU, area absolute error (mm²), centroid distance, failure rate and inter-rater variability; never silently omit non-detections.
- [ ] Evaluate SVC and DVC slab contamination, residual superficial projection artifacts, banding, shadow attenuation and nonperfusion false positives.
- [ ] Evaluate spatial calibration and uncertainty after any resizing or decimation. Preview mode should never be pooled as quantitative full-data measurements.
- [ ] Repeat scans to measure repeatability CV/ICC and examine protocol/center/vendor/ethnicity/disease subgroup shift where appropriate.
- [ ] Track calibration and algorithm settings in each export. Preserve original images and user-revised segmentation annotations separately.
- [ ] Establish usage safeguards, access control, encrypted storage and auditable data retention before deploying to a shared environment.

## Roadmap

1. Vendor adapters (DICOM OCT, Heidelberg `.vol`, `.e2e` only when calibration is trusted) and device normalization with coordinate transforms.
2. Pluggable trained segmentation backends with calibrated confidence estimates and GPU/CPU fallback. Optional Choroidalyzer integration with explicitly provided weights and independent verification.
3. Editable manual annotation of fovea/FAZ/boundaries with versioned revisions and review sign-off.
4. True flow-estimation support only when repeated registered structural acquisitions and validated algorithms are available; otherwise analyze precomputed OCTA flow input.
5. Multi-scale perfusion density, vessel length density with diagonal-length correction, annular ETDRS regional analysis, tortuosity, fractal dimension, intercapillary area metrics, choroidal vascularity index, choriocapillaris flow deficits.
6. Batch queues and worker pools (Celery/RQ), GPU scheduling, resumable uploads, deduplication, quotas and scoped identity/access management.
7. End-to-end regression datasets and snapshot-based visual verification for representative pathologies and scanner vendors.

## Methodological non-claims

- The current default projection is **not** a calibrated OCTA decorrelation reconstruction from raw interference data.
- The outer retina/choroid panels are **experimental proxy slabs**, not validated anatomical segmentation.
- A measured OCTA signal does not itself give absolute blood-flow velocity.
- A passed QC status only means the implemented heuristics did not flag the scan.
- FAZ area in pixels cannot be converted to mm² without verified lateral acquisition dimensions.
