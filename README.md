# Darbandi FDK Reconstruction

Darbandi FDK Reconstruction is a desktop application for cone-beam micro-CT reconstruction using CERN/TIGRE FDK. It is designed for Zeiss-style workflows where `.txrm` data has already been extracted into TIFF projections, TIFF reference images, and a projection-geometry CSV.

The app provides:

- metadata validation and projection preview
- averaged flat-field correction
- attenuation conversion with configurable clipping
- Zeiss-style per-projection shift correction
- center-offset preview search and automatic metrics
- TIGRE FDK reconstruction
- image display, zoom, plots, logs, config save/load, and TIFF output

## Quick Start

Create the recommended TIGRE environment:

```powershell
conda env create -f microct_tigre_gui/environment-tigre.yml
conda activate tomogram-recon-tigre
```

Run the app:

```powershell
python fdk-engine.py
```

Run a lightweight non-GUI self-test:

```powershell
python microct_tigre_gui/main.py --self-test
```

For installation details, data preparation, reconstruction workflow, troubleshooting, and output descriptions, see [USER_MANUAL.md](USER_MANUAL.md).

## Expected Input Layout

```text
dataset/
  projections/
    proj_000001.tif
    proj_000002.tif
  reference/
    reference_001.tif
  metadata/
    projection_geometry.csv
```

The CSV should include a projection filename column and an angle column. Optional `angle_rad`, `x_shift_px`, and `y_shift_px` columns are supported.

## Important TIGRE Note

Do not install the unrelated PyPI package named `tigre`. This app requires the CERN CT reconstruction toolbox. The recommended route is the conda package from `ccpi`, as listed in `environment-tigre.yml`.

## Repository Contents

```text
fdk-engine.py                         launcher
microct_tigre_gui/main.py             CLI/self-test entry point
microct_tigre_gui/src/                GUI and reconstruction source
microct_tigre_gui/config_example.json example GUI configuration
microct_tigre_gui/environment-tigre.yml recommended conda environment
USER_MANUAL.md                        full user manual
```

## License

No open-source license has been added yet. Until a license is added, the default copyright rules apply.
