# Analysis

## Description

This is the analysis program that can analyze a model and provide the user with details on the memory and computation cost of a provided update scheme. Furthermore, it includes a scheme-searcher component, which can be used to generate the best update scheme for a given model with user defined constraints and optimization parameters.

## Setup

> TODO - requirements for python script

## Usage

### Update PYTHONPATH

```bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)
```

### Pre-processing

```bash
python analysis/pre-processing/combine_jsons.py
```

### Memory and computation cost analysis

```bash
cargo run analysis
```
