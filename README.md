# Energy aware on device learning on microcontrollers

## Setup

> TODO - requirements

## Usage

### Memory and computational cost analysis

```bash
# Update python path
export PYTHONPATH=${PYTHONPATH}:$(pwd)
# Preprocessing of data
python analysis/pre-processing/combine_jsons.py
# Analysis
cargo run analysis
```
