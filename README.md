# Forest-cover transition inventories in Baoxing County

This repository contains the analysis scripts for comparing three and five consecutive years of source and destination land cover. It covers mapped transition footprints and retrospective spatial prediction of net primary productivity. The code is paired with Supplementary Data 1 prepared for the manuscript.

## Data sources and access

| Product | Data used | How to obtain it |
| --- | --- | --- |
| China Land Cover Dataset 1.0.5 | Annual 30 m land-cover maps, 2000–2025 | Download the specified annual rasters from the [fixed Zenodo release](https://doi.org/10.5281/zenodo.18180184). |
| MOD17A3HGF.061 | Annual NPP and gap-fill percentage, tile h26v05, 2001–2005 and 2020–2024 | Open the [NASA product record](https://doi.org/10.5067/MODIS/MOD17A3HGF.061) and select the tile and years in Earthdata. |
| ERA5-Land | Monthly mean 2 m air temperature | Select monthly averaged reanalysis, 2 m temperature and the study years in the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means). |
| CHIRPS v2.0 | Annual precipitation totals, 0.05° | Use the version 2.0 global annual files distributed through the [Climate Hazards Center](https://www.chc.ucsb.edu/data/chirps). |
| Copernicus DEM GLO-30 | Surface elevation and derived slope | Select GLO-30 for the study area in the [Copernicus Data Space](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM). |

The administrative boundary used in the analysis was supplied by the local natural resources authority for the 2025 land-change survey. Access to the original boundary requires permission from the supplying authority. Using a separately licensed boundary can change intersection areas.

## Reproducing the results

Use the anonymized tables in the manuscript's companion file `Supplementary_Data_1.zip`. Extract its `data` directory alongside `code`. Run the following from the repository root with NumPy 2.2.0 and pandas 2.3.0.

```sh
python code/inventory.py --output reproduced
python code/replay.py --output reproduced
```

The first command reconstructs mapped directional footprints from frequencies of class histories. The second reproduces pooled held-out errors, the sensitivity analysis with predictor observations through 2019, conditional block-resampling summaries, positive-cell correlations and gap-fill summaries. To reproduce a single primary additive fit, use R 4.6.1 with mgcv 1.9-4.

```sh
Rscript code/refit.R retrospective main GAM B M3 1 fitted_values.csv
```

The analytical supplement uses anonymous identifiers. Source products are distributed by the original providers under their respective access terms. The original boundary, native imagery and cell locations are excluded from this repository.
