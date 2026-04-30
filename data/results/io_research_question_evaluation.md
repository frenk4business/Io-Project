# Io Multi-Metric Volcanism Research Question Evaluation

## Final Classification

**Answered as an exploratory analysis.**

The implementation supports a reproducible, hypothesis-generating comparison of named hotspot occurrence, estimated thermal intensity, and metadata-normalized observation activity on a common 1 degree grid. It does not support a near-publication-grade claim of true coverage correction because real footprints, sensitivity modeling, and systematic non-detections are not included.

## Research Question

**To what extent do different volcanic activity metrics, including named hotspot occurrence, estimated thermal intensity, and metadata-normalized observation activity, produce different spatial interpretations of Io's volcanism on a common 1 degree grid?**

## Direct Answer

The current analysis shows that Io's apparent volcanic activity pattern is metric-dependent. Event occurrence and metadata-normalized intensity differ strongly in the current result files: Spearman correlation is `-0.424`, top-10% rank overlap is `0.000`, and Jensen-Shannon divergence is `0.294`. Estimated Davies/JIRAM proxy power is concentrated, with top-10/top-25/top-50 cells contributing `35.3%`, `55.7%`, and `73.3%` of the estimated proxy GW layer.

## Evidence Table

| Component | Status | Supporting file/function/output | Reviewer comment |
|---|---|---|---|
| Common 1 degree grid | Present | `cell_id`, `lon_centre`, `lat_centre` across metric layers | Metrics are spatially comparable on the shared grid. |
| Named hotspot occurrence | Present | `has_hotspot`, `hotspot_count` | Separated from thermal event occurrence. |
| Event occurrence | Present | `occurrence_event_count` | Includes normalized event rows from available thermal sources. |
| Estimated thermal intensity | Present | `radiant_power_gw_layer`, radiance/brightness layers, normalized intensity proxies | Physical/semi-physical proxy GW is kept separate from unitless percentile proxies. |
| Metadata-normalized activity | Present but limited | `observation_count`, `coverage_weight`, `coverage_corrected_event_rate`, `coverage_corrected_intensity` | Metadata-based normalization only; not true footprint/sensitivity correction. |
| Quantitative comparison | Present | Spearman, top-10% overlap, JS divergence, latitude bands, top-N curves | Supports metric-dependent spatial interpretation. |
| Persistence/episodicity | Limited | `activity_class`, `persistence_score` | Requires better independent observation windows before strong persistence claims. |

## Current Data Quality

- Activity event rows: `2386`
- Metadata coverage rows: `1351`
- Metadata covered cells: `1108`
- Metadata coverage instruments: `JIRAM, KECK/GEMINI AO, NIMS`
- Nonzero metadata-normalized intensity cells: `1088`

## Needed Improvements

- Add real JIRAM/NIMS/AO footprints, observation windows with non-detections, and sensitivity/geometry masks for publication-grade coverage correction.
- Keep Davies/JIRAM estimated proxy GW separate from radiance, brightness, and unitless percentile-normalized proxies.
- Treat persistence and episodicity as provisional until independent observation windows are available.
- Use dashboard language such as metadata-normalized activity or metadata-based coverage normalization, not true coverage-corrected volcanism.

## Suggested Scientific Claim

**Cautious supported claim:** This analysis shows that Io's apparent volcanic activity pattern is metric-dependent. On a common 1 degree grid, named hotspot occurrence, estimated thermal intensity, and metadata-normalized activity can produce different spatial rankings, indicating that hotspot catalogs alone do not fully represent the spatial structure of observed thermal activity.

**Stronger supported addendum:** Estimated Davies/JIRAM proxy power appears more concentrated than hotspot occurrence, with the top 10 power cells contributing about `35%` and the top 50 contributing about `73%` of the estimated proxy GW signal.
