# Kaggle Campaigns Inventory

Source dataset: `https://www.kaggle.com/datasets/maramirezes/campaigns`

Imported on: `2026-03-18`

## Placement

- Raw, untouched source files live in `data/raw/campaigns/`.
- Dataset-level descriptive files copied from the source package live in this directory.

## Extracted Contents

- Total extracted size: about `2.6G`
- Total files: `386`
- Campaign directories: `53`
- Campaign metadata files: `53`
- Node measurement CSV files: `331`

Root-level source files:

- `README.md`
- `.gitignore`

## Campaign Families

- `ANTENNA_sweep`: `1` campaign directory
- `LNA16_VGA*`: `4` campaign directories
- `LNA_sweep_*`: `6` campaign directories
- `RBW_sweep_*`: `6` campaign directories
- `VGA_sweep_*`: `32` campaign directories
- `fm_ref_fullband_*`: `2` campaign directories
- `fm_sweep_rbw_*`: `2` campaign directories

## File Pattern

Each campaign directory contains:

- `metadata.csv` with campaign-level acquisition settings
- `Node*.csv` files with node-level PSD acquisitions

Observed node-file distribution across campaign directories:

- `43` campaigns with `6` node CSV files
- `7` campaigns with `7` node CSV files
- `3` campaigns with `8` node CSV files

## Observed Schemas

Campaign metadata columns:

- `campaign_label`
- `campaign_id`
- `start_date`
- `stop_date`
- `start_time`
- `stop_time`
- `acquisition_freq_minutes`
- `central_freq_MHz`
- `span_MHz`
- `sample_rate_hz`
- `lna_gain_dB`
- `vga_gain_dB`
- `rbw_kHz`
- `antenna_amp`

Node measurement columns:

- `id`
- `mac`
- `campaign_id`
- `pxx`
- `start_freq_hz`
- `end_freq_hz`
- `timestamp`
- `lat`
- `lng`
- `excursion_peak_to_peak_hz`
- `excursion_peak_deviation_hz`
- `excursion_rms_deviation_hz`
- `depth_peak_to_peak`
- `depth_peak_deviation`
- `depth_rms_deviation`
- `created_at`
