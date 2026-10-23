"""Tunable parameters for the OCTA reconstruction pipeline.

All depth offsets are expressed in **pixels relative to the ILM boundary**,
calibrated empirically from the ILM-aligned mean intensity profile (see
notebooks/dev notes). The pipeline does not currently know the true axial
(um/pixel) resolution of the acquisition device -- this was not provided
with the dataset. If Remidio supplies the axial sampling interval, convert
the pixel offsets below to physical units (um) so the slab definitions can
be pinned to the standard clinical plexus definitions:

    Superficial (SVP): ILM            -> IPL/INL junction
    Deep        (DVP): IPL/INL junction -> ~outer plexiform / ONL boundary

instead of the current pixel-offset proxies.
"""

from dataclasses import dataclass, field


@dataclass
class SegmentationConfig:
    # ---- noise floor / ILM detection ----
    ilm_noise_rows: int = 40          # rows from the very top assumed to be vitreous/background
    ilm_threshold_k: float = 4.0      # threshold = noise_floor + k * noise_std
    ilm_sustain_rows: int = 15        # min consecutive rows above threshold to accept as ILM
    ilm_guard_rows: int = 3           # skip a few px below ILM (vitreoretinal interface noise)
    ilm_persist_rows: int = 80        # a genuine ILM stays bright for a long trailing window (real
                                       # retina, all the way toward RPE); a transient artifact (a
                                       # floater, a specular reflection) does not. A candidate crossing
                                       # must also have at least `ilm_persist_frac` of this trailing
                                       # window above threshold to be accepted -- otherwise it's
                                       # rejected and the search continues for the next candidate.
    ilm_persist_frac: float = 0.6

    # ---- low-confidence column detection (see segmentation._detect_low_confidence_columns) ----
    # Catches the case where a column has no real inner-retina signal at
    # all -- the coarse ILM detector then latches onto some deeper bright
    # structure (which genuinely does satisfy "sustained and persistent"),
    # silently producing a wrong ILM and a wrong everything built on it.
    # Checked at a FIXED window relative to the B-scan's *median* ILM
    # (robust to a few bad columns), not each column's own possibly-wrong
    # ILM -- checking against a column's own detection would be circular.
    confidence_window: tuple = (-20, 80)   # rows relative to the B-scan's median ILM
    confidence_threshold_k: float = 5.0    # threshold = noise_floor + k * noise_std
    # The peak check's window edges can still carry a weak, brief brightness
    # bump from the ramp into/out of a real low-signal region -- enough to
    # clear the threshold even though the ILM found there is still wrong
    # (found by direct inspection: a column right outside the flagged core
    # still had its ILM sitting ~135px deeper than normal). Rather than
    # chase this by tuning the threshold -- tried, and no single value
    # cleanly separated the two patients' cases -- the flagged core is
    # grown by this many columns on each side before bridging, as a safety
    # margin. This is safe to make generous: on a patient with no flagged
    # core at all (a real, correctly-detected foveal pit), dilating an
    # empty set is still empty, so this cannot affect genuine anatomy --
    # only widens coverage where a real problem was already found.
    confidence_dilate_px: int = 45

    # ---- global smoothing ----
    # A lighter version of this (sigma 1.5, median width 15) was tried,
    # reasoning that the DP shortest-path traces already provide the main
    # smoothness guarantee within each B-scan so heavy post-hoc smoothing
    # might just be hiding real detail. It was checked against QC outlier
    # fractions (barely moved) and against Patient 3/5's boundary overlays
    # (stayed clean) before being adopted -- but not against Patient 4's
    # B-scan 224, which is exactly where it mattered: with the lighter
    # median filter, the boundaries plunge almost to RPE across the same
    # signal-void region Bug 3 already bridges, instead of the gentle,
    # anatomically-plausible double-dip the heavier smoothing produces
    # there. Checking QC *statistics* is not the same as looking at the
    # specific known-hard region, and this is why the check has to include
    # both. Reverted to these values, which is what every case in this
    # dataset (including the Patient 4/5 depression fix) was actually
    # validated against.
    gauss_sigma_bscan: tuple = (2.5, 2.5)   # (depth, width) sigma for per-B-scan smoothing (segmentation only)
    boundary_median_width: int = 29          # across A-scans (x); wide enough to reject narrow
                                              # vessel-shadow-artifact spikes in the boundary trace
    boundary_gauss_sigma_xy: tuple = (3.0, 3.0)  # smoothing the boundary surface across (B-scan, A-scan)

    # ---- ILM-relative offset windows (pixels) ----
    # Retinal thickness (hence the true ILM-relative offset of every deeper
    # band) varies a lot across a volume -- it is markedly thinner in the
    # periphery than near the fovea, and pathology (edema, fluid, PED) can
    # distort it further. Fixed absolute-pixel windows calibrated on one
    # region of one B-scan were found to misfire elsewhere in the volume,
    # so only IPL/INL uses a fixed search window (it directly follows the
    # first, always-present RNFL/GCL/IPL bright complex, which sits close
    # to the ILM regardless of overall retinal thickness). RPE and
    # OPL/ONL use scale-invariant strategies instead -- see segmentation.py.
    dmax: int = 320
    ipl_inl_window: tuple = (35, 140)   # IPL/INL junction (first valley after the first peak)
    rpe_min_offset: int = 25            # skip this far past ILM before searching for RPE (avoids
                                         # the RNFL/GCL/IPL peak); RPE = global brightest point after this
    opl_onl_fraction: float = 0.35      # OPL/ONL boundary, as a fraction of the IPL/INL -> RPE
                                         # span (approximates INL+OPL thickness proportion; see README)

    # ---- slab margins ----
    svp_top_guard: int = 3     # px below ILM to start the superficial slab (skip vitreoretinal noise)
    dvp_bottom_extra: int = 0  # extra px added below the OPL/ONL junction for the deep slab

    # ---- dynamic-programming boundary tracing ----
    # Every boundary is found as a smoothness-constrained shortest path
    # across the whole B-scan (see segmentation._dp_trace), not an
    # independent per-A-scan search -- this is what keeps the traces clean
    # under speckle noise, vessel shadows and pathology instead of jagged.
    # `max_step` bounds how many pixels the boundary may move between
    # adjacent A-scans; `step_penalty` further discourages movement even
    # within that bound (0 = free movement up to max_step).
    ilm_dp_band: int = 25         # ILM: +/- px around the coarse threshold estimate to refine within
    ilm_dp_max_step: int = 3
    ilm_dp_step_penalty: float = 0.0
    ipl_inl_dp_max_step: int = 3
    ipl_inl_dp_step_penalty: float = 0.0
    ipl_inl_depth_bias: float = 1.5   # cost added per px of depth within the window; see segmentation.py
    rpe_dp_max_step: int = 4
    rpe_dp_step_penalty: float = 0.0
    # A DP-refined, image-derived OPL/ONL boundary (fraction anchor + narrow-
    # band gradient search) was implemented and tuned to be numerically
    # stable, but reverted after visual inspection showed it converging on
    # the wrong edge (likely ELM/IS-OS) instead of the true OPL/ONL junction
    # -- see the detailed rejection note in segmentation.py at the
    # `opl_onl = ipl_inl + cfg.opl_onl_fraction * ...` line, and README
    # "Known limitations" for what a validated fix would actually need.

    # ---- robustness ----
    max_frame_to_frame_jump: int = 40  # px; outlier rejection along the B-scan axis
    boundary_outlier_y_window: int = 21  # B-scans; window for the robust y-direction reference (odd)
    edge_guard_cols: int = 6           # flatten this many columns at each A-scan-edge (signal roll-off)

    # ---- automated QC thresholds (see segmentation._check_surface_quality) ----
    # These exist so a bad segmentation announces itself (a logged warning +
    # a `flagged` field in the saved metrics/QC report) instead of silently
    # shipping a wrong enface image -- essential once this runs unattended
    # on volumes nobody is going to eyeball one by one.
    qc_max_outlier_fraction: float = 0.02       # >2% of columns needing y-outlier correction is suspicious
    qc_max_residual_jump_px: float = 25.0       # max allowed frame-to-frame jump after regularization
    qc_min_slab_thickness_px: float = 3.0       # SVP/DVP slab thinner than this (px) is implausible
    qc_max_slab_thickness_px: float = 300.0     # SVP/DVP slab thicker than this (px) is implausible


@dataclass
class ProjectionConfig:
    # OCTA flow/decorrelation signal is sparse (bright speckles on a near-
    # zero background), so a max-intensity projection within each slab
    # preserves the vessel signal much better than a mean projection would
    # (mean dilutes it with the many empty-background A-scan samples in the
    # slab) -- this matches standard commercial OCTA enface generation.
    projection_mode: str = "max"        # "mean" or "max"
    projection_artifact_suppression: float = 0.35  # fraction of the superficial peak modeled as shadow
    # Depth-resolved (3D) projection-artifact suppression -- see
    # projection.suppress_projection_artifact_3d for why this replaced an
    # earlier enface-level (2D, post-projection) version: for a MAX
    # projection, a flat per-A-scan subtraction is mathematically identical
    # whether it's applied before or after projecting, so the real fix
    # isn't *when* the subtraction happens but making it decay with depth
    # instead of being flat, so a true independent deep vessel farther from
    # the superficial peak can survive it. Not a calibrated physical
    # distance (axial um/pixel wasn't supplied with this dataset) -- tuned
    # visually.
    projection_artifact_decay_px: float = 20.0
    flow_median_ksize: int = 3          # despeckle flow volume before projection

    # Bulk transverse motion during a B-scan's acquisition inflates flow
    # signal across the whole frame, not just true vessels, producing a
    # bright horizontal line in the enface projection. Detected via the
    # ONL band (real tissue, normally avascular) and corrected by
    # subtracting each B-scan's excess "floor" over the cohort baseline.
    # See projection.estimate_bulk_motion_floor / suppress_bulk_motion.
    bulk_motion_ref_margin: int = 10           # px guard from OPL/ONL and RPE for the ONL reference window
    bulk_motion_baseline_percentile: float = 50.0  # percentile treated as the "normal" (uncorrupted) floor


@dataclass
class VisualizationConfig:
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: tuple = (8, 8)
    vessel_density_threshold: float = 0.5  # fraction of Otsu threshold used for vessel-density binarization

    # Multiscale vesselness (Frangi) blended into a separate "vessel-
    # enhanced" output on top of the existing CLAHE-enhanced one -- see
    # postprocessing.enhance_vesselness for why this exists and why it's
    # applied uniformly to every patient rather than selectively.
    vesselness_sigmas: tuple = (1, 2, 3)  # capillary-to-small-vessel scale range, in px
    vesselness_blend: float = 0.4          # 0 = pure CLAHE, 1 = pure vesselness map

    # Display-only brightness calibration -- see postprocessing.apply_display_gamma.
    # Target median calibrated against a reference high-quality wide-field
    # OCTA image (median ~41/255); not applied to the file used for vessel-
    # density thresholding or the external validation model.
    display_gamma_target_median: float = 40.0
    display_gamma_min: float = 0.35  # floor on how aggressively an already-dim image can be brightened


@dataclass
class PipelineConfig:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
