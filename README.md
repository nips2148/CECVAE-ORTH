CECVAE-ORTH
==
This is the code for implementing the CECVAE-ORTH proposed by "Disentangled Representation Learning with Variational Inference: A Method for Treatment Effect Estimation" (NeurIPS 2021 Conference Paper2148). 

It is written in python 3.7 with numpy 1.18.1 and tensorflow 1.13.1.

The code of CECVAE-ORTH is built upon the Counterfactual regression (CFR) work of Johansson, Shalit & Sontag (2016) and Shalit, Johansson & Sontag (2016), https://github.com/clinicalml/cfrnet.
The parameter searching, network training and evaluation follow the procedures of CFR to ensure fair comparison.

To run parameter search:

```python cvae_param_search.py <config_file> <num_runs>```

To evaluate the results:

```python evaluate.py <config_file> [overwrite] [filters]```

Dataset
-
Both the IHDP and Jobs dataset are avaliable from https://www.fredjo.com/

Raw data of the TWINS is taken from here:
http://www.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html
Specifically these files:
http://www.nber.org/lbid/1989/linkco1989us_den.csv.zip,
http://www.nber.org/lbid/1990/linkco1990us_den.csv.zip,
http://www.nber.org/lbid/1991/linkco1991us_den.csv.zip

The dataset guide is available here:
http://www.nber.org/lbid/docs/LinkCO89Guide.pdf

Example
-
To run parameter search:
```python drnet_param_search.py configs/example_Jobs.txt 10```

To evaluate the results:
```python evaluate.py configs/example_jobs.txt 1```