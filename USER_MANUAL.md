# User Manual: Darbandi FDK Reconstruction

This manual explains how to install, configure, run, and troubleshoot the Darbandi FDK Reconstruction app.

## 1. Purpose

Darbandi FDK Reconstruction is a graphical Python application for micro-CT FDK reconstruction using CERN/TIGRE. It is intended for Zeiss-style cone-beam CT data that has already been exported into:

- TIFF projection images
- TIFF reference / flat-field images
- a metadata CSV containing projection filenames and angles

The app handles preprocessing, geometry setup, center-offset search, per-projection shift correction, TIGRE FDK reconstruction, display, plotting, and logs.

## 2. Installation

### Recommended Conda Environment

Install Anaconda or Miniconda, then create the TIGRE environment:

```powershell
conda env create -f microct_tigre_gui/environment-tigre.yml
conda activate tomogram-recon-tigre
```

This environment installs Python, NumPy, SciPy, pandas, tifffile, Matplotlib, OpenCV, PyQt, and TIGRE from conda channels.

### Alternative GUI-Only Install

If you only need to inspect metadata, preprocess, or run the self-test without TIGRE FDK:

```powershell
python -m pip install -r microct_tigre_gui/requirements.txt
```

TIGRE reconstruction will not run until CERN/TIGRE is installed.

## 3. Launching the App

From the repository root:

```powershell
python fdk-engine.py
```

To test core non-GUI logic:

```powershell
python microct_tigre_gui/main.py --self-test
```

At startup, the app logs whether TIGRE is importable. If TIGRE is missing, preprocessing and plotting still work, but FDK reconstruction will show an error.

## 4. Required Data Layout

Use a folder layout like:

```text
dataset/
  projections/
    proj_000001.tif
    proj_000002.tif
    ...
  reference/
    reference_001.tif
    reference_002.tif
    ...
  metadata/
    projection_geometry.csv
```

The metadata CSV should contain:

- projection filename column, for example `tiff_file`
- angle column in degrees, for example `angle_deg`
- optional radian angle column, for example `angle_rad`
- optional per-projection shift columns, for example `x_shift_px` and `y_shift_px`

## 5. Basic Workflow

1. Select the projection folder.
2. Select the reference / flat-field folder.
3. Select the metadata CSV or metadata folder.
4. Select an output folder.
5. Click `Load Metadata CSV`.
6. Confirm filename, angle, radian-angle, and optional shift columns.
7. Click `Validate Metadata`.
8. Click `Compute Averaged Flat Field`.
9. Click `Apply Flat-Field Correction`.
10. Click `Compute Attenuation Projections`.
11. Enter geometry values.
12. Optionally run center-offset search or shift-convention tests.
13. Click `Run FDK Reconstruction`.
14. Save or inspect outputs in the output folder.

## 6. Optimized Default Settings

The current GUI defaults are tuned for the optimized Zeiss-style workflow:

- `Remove duplicate 0/360 endpoint`: enabled
- `Invert angle sign for TIGRE`: enabled
- `Clip transmission`: enabled
- `Set negative attenuation to zero`: enabled
- `Enable per-projection shift correction`: enabled
- shift stage: before `-ln`, on the transmission stack
- shift preset: Mode A
- x shift sign: `+1`
- y shift sign: `+1`
- shift interpretation: apply shifts as correction values

These defaults can still be changed manually or by loading a saved config file.

## 7. Geometry Inputs

The TIGRE geometry panel asks for:

- effective pixel size at object plane, in micrometers
- source-to-object distance, DSO, in millimeters
- source-to-detector distance, DSD, in millimeters
- voxel size, in micrometers
- reconstruction size `Nx`, `Ny`, `Nz`
- optional object offsets
- optional detector offsets

The app computes TIGRE detector pixel size as:

```text
d_detector = effective_pixel_size_mm * DSD / DSO
```

Use millimeters for DSO/DSD and micrometers for pixel/voxel size.

## 8. Preprocessing

The app assumes no dark-current image is available. Flat-field correction is:

```text
T = I / F
```

where `I` is each projection and `F` is the averaged reference / flat-field image.

Attenuation is:

```text
p = -ln(T)
```

The clipping controls prevent unstable logarithms and suppress negative attenuation when selected.

## 9. Per-Projection Shift Correction

If the metadata has `x_shift_px` and `y_shift_px`, the app can apply per-projection shifts before attenuation conversion.

The default optimized mode is:

```text
row_shift = +1 * y_shift_px
col_shift = +1 * x_shift_px
```

This is Mode A. Other modes are available for sign testing.

Use `Run Shift-Convention Test` to reconstruct small debug volumes for all sign modes and compare central slices.

## 10. Center-Offset Search

The `Center Offset Search` panel helps find the detector horizontal center offset before full reconstruction.

Controls:

- start shift, in pixels
- end shift, in pixels
- step, in pixels
- preview slice index
- single-slice or thin-band preview mode
- center-shift sign convention
- manual preview search
- automatic search
- preview slider
- apply selected shift
- use automatic best shift

Manual search reconstructs a preview for every shift candidate. Automatic search also calculates:

- gradient energy
- Laplacian variance
- Shannon entropy
- combined score

After inspecting the previews, click `Apply Selected Shift` or `Use Automatic Best Shift` to copy the chosen value into the manual center-offset field.

## 11. Image Display

The image display panel can show:

- raw projection
- averaged flat field
- transmission projection
- attenuation projection
- reconstructed Z slice
- center-offset preview

Use `Zoom In`, `Zoom Out`, and `Reset Zoom` to inspect the current displayed image. Zoom changes only the display view, not the data.

## 12. Outputs

When an output folder is selected, the app may write:

```text
averaged_flat_field.tif
reconstruction.tif
reconstruction_stack/
processing_log.txt
geometry_sanity_report.txt
center_offset_search_metrics.csv
shift_debug/
shift_mode_tests/
transmission_stack/
attenuation_stack/
```

Transmission and attenuation stacks are saved only if their checkboxes are enabled.

## 13. Saving and Loading Configurations

Use `Save Config` to write current GUI settings to JSON.

Use `Load Config` to restore a previous reconstruction setup.

Use `Reset Parameters` to return to the built-in defaults.

The file `microct_tigre_gui/config_example.json` shows the supported config fields.

## 14. Troubleshooting

### TIGRE Is Not Importable

Install the recommended conda environment:

```powershell
conda env create -f microct_tigre_gui/environment-tigre.yml
conda activate tomogram-recon-tigre
```

Do not install the unrelated PyPI package named `tigre`.

### Native Runtime Abort / `forrtl: error (200)`

This usually means the native TIGRE runtime was interrupted by Ctrl+C, an IDE stop button, closing the terminal, or killing the process during FDK. Avoid interrupting while TIGRE is inside native/CUDA code.

For large datasets, start with smaller tests:

- use 2x or 4x binning
- reduce `Nx`, `Ny`, and `Nz`
- run center-offset preview before a full volume
- use fewer shift candidates

### Wrong Rotation Direction

Try toggling `Invert angle sign for TIGRE`.

### Duplicate 0/360 Projection

Keep `Remove duplicate 0/360 endpoint` enabled when the metadata contains both endpoints.

### Poor Centering

Use the center-offset search panel. If artifacts worsen in the expected direction, flip `Center-shift sign convention`.

### Large Memory Use

Avoid saving intermediate stacks unless needed. Use binning and smaller debug volumes first.

## 15. Reproducibility

For each serious reconstruction, save:

- config JSON
- processing log
- geometry sanity report
- center-offset metric CSV, if automatic center search was used

These files document angle handling, shifts, geometry, offsets, filter choice, and output paths.
