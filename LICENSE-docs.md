# License for the documentation and content

The **code** in this repository is under the MIT License (see `LICENSE`).

The **documentation and content** are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
That covers `README.md`, `DOCUMENTATION.md`, the scenario designs in `metric_clinic_presets.py`
and `example_scenarios.json`, and the simulation design they describe.

You may share and adapt this material for any purpose, including commercially, **as long as you
give appropriate credit**, link to the license, and say whether you made changes. Suggested credit:

> Model Selection Clinic simulator by Dean Abbott, Abbott Analytics
> (https://github.com/abbottanalytics/abbott-error-metric-clinic), CC BY 4.0.

Copyright (c) 2026 Dean Abbott, Abbott Analytics.

## Data

The real-model files in `data/` hold predictions from models trained on the
**KDD Cup 1998** donor data (Paralyzed Veterans of America), distributed by the UCI KDD Archive:
https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html. Only the targets of the held-out records
(TARGET_B, TARGET_D) and model scores are included; no donor inputs.
