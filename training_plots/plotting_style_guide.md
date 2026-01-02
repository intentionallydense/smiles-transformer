# Plotting Style Guide

This guide defines the visual standards for plotting training results to ensure consistency across the project.

## General Guidelines
- **Library:** Use `matplotlib` (and `seaborn` for aesthetics if available, otherwise stick to standard `matplotlib` styling).
- **Figure Size:** Standard figure size should be `(10, 6)` inches.
- **Resolution:** Save figures with `dpi=300` for high quality.
- **Font:** Use a sans-serif font (e.g., Arial or Helvetica, or the default sans-serif).
- **Grid:** Always include a light grid for readability (`alpha=0.3`).

## Colors
- **Training Data:** Blue (e.g., `#1f77b4` or 'b')
- **Validation Data:** Orange (e.g., `#ff7f0e` or 'orange')
- **Baseline/Comparison:** Green or Red.

## Line Styles
- **Training Loss:** Solid line (`-`), linewidth 2.
- **Validation Loss:** Dashed line (`--`), linewidth 2.
- **Markers:** Use markers for data points (e.g., 'o') with a size of 5.

## Labels and Titles
- **Title:** Bold, Font size 14. Format: "Training and Validation Loss - [Run Name]"
- **Axis Labels:** Font size 12. Explicitly state metric and unit (if applicable).
  - X-axis: "Epoch"
  - Y-axis: "Loss"
- **Legend:** Include a legend located in the 'best' position. Font size 10.

## Output
- **File Format:** PNG
- **Naming Convention:** `[run_name]_loss_plot.png`
