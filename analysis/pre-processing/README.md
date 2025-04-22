# Pre-processing

This folder contains all the files needed to produce JSONs corresponding to each model and the pre-processing script `combine_jsons.py`.

## `analysis_data`

Contains information regarding how much updating each layer impacts the total downstream accuracy.

## `json_base`

JSON representations of the models each with different weight update ratios.

## `combine_jsons.py`

This is the pre-processing script that outputs one `JSON` file with all relevant model information to perform the offline analysis and identify the best update scheme
