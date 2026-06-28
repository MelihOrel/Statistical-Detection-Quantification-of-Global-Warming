# data/raw

This folder ships with `GlobalTemperatures.csv` (the global monthly series).

The regional steps additionally use two Berkeley Earth companion files, which
are large and are NOT bundled here. Download them from the Kaggle
"Climate Change: Earth Surface Temperature Data" dataset and place them here:

- GlobalLandTemperaturesByCountry.csv
- GlobalLandTemperaturesByMajorCity.csv

With them present, `python main.py` produces the country ranking and the
Turkey case study automatically. Without them, those steps log a notice and
the rest of the pipeline still runs end to end.
