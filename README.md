# Evaluation of Global Descriptors for Visual Localization

This is a fork from the [`hloc`](https://github.com/cvg/Hierarchical-Localization) repo for the evaluation of the 
[DnS](https://arxiv.org/abs/2106.13266) and [Geolocation](https://dl.acm.org/doi/10.1145/3460426.3463644) methods 
as global image descriptors for Visual Localization on the Aachen Day-Night dataset.

## Installation

This requires Python ==3.8 and PyTorch >=1.1. Installing the package locally pulls the other dependencies:

```bash
git clone --recursive https://github.com/gkordo/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
```

## Run the code

* Run the following script by providing the method's name, i.e. `dns`, `geoloc` or `netvlad`, input image size and 
refinement, i.e. `none`, `W`, `MS`, `QE` or `TD`:

```bash
bash scripts/run_method.sh <output_path> <method_name> <im_size> <refinement>
```

* Run a method for all image sizes with no refinements:

```bash
bash scripts/run_method_all_sizes.sh <output_path> <method_name>
```

* Run a method with the refinements for a given image size:

```bash
bash scripts/run_method_all_refinements.sh <output_path> <method_name> <im_size>
```

* Run all methods with the all refinements for a given image size:

```bash
bash scripts/run_method_all_refinements.sh <output_path> <method_name> <im_size>
```