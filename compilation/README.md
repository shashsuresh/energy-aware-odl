# TINY-TRAINING-FRAMEWORK

> This repository is a condensed version of MIT HAN Lab's [tiny-training repository](https://github.com/mit-han-lab/tiny-training/tree/main), containing refactored versions of the `Quantization` and `Compilation` code, along with a few mini modules to facilitate the conversions.

## Overview

What is the purpose of this repo?

## Pre-requisites

What does one need to have installed?

## Structure

The repository is split up into two main directories

`mcu_conversion` - Contains all modules pertaining to conversion of PyTorch models into JSON files, that will form the basis of the code that is going to be deployed on MCUs

`quantization` - Contains a collection of quantized versions of layers found in DNNs along with some utility functions. These are used for conversion and also to simulate MCU inference on PCs and GPUs.

## Generating models for deployment

### Set up environment

```bash
export QUANT=~/TUDelft/thesis/tiny-training-custom/quantization
export PYTHONPATH=$QUANT:${PYTHONPATH}
export TVM_HOME=~/TUDelft/thesis/tvm-hack
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

### Generate IRs

Define all your configs in the `dict`

> python mcu_conversion/ir_generation

### Generate JSONs

> python mcu_conversion/json_conversion

### Single interface - TODO

TODO based on our new interface - maybe add a GUI to visualize config

## Next Steps

TODO some info about **TTE**

## Testing

Test folder has a set of tests and test_refs that can be used to test the IR generation

To test:

```bash
python -m unittest discover -s tests -v
```
