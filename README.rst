===============================
TSGNet
===============================

This is an implementation of TSGNet:  Exploiting Interaction Links for Node Classification with Deep Graph Neural Networks, IJCAI 2019

Usage
-----

**Example Usage**
    ``$main.py --dataname "imdb"


**Full Command List**
    The full list of command line options is available with ``$main.py --help``


Requirements
------------
We are tested in the following environment, but it may work in their previous versions.
* Python 3
# Pytorch 1.7
* Python geometric




Citing
------
If you find DeepWalk useful in your research, we ask that you cite the following paper::

@inproceedings{parkneville2019,
  title     = {Exploiting Interaction Links for Node Classification with Deep Graph Neural Networks},
  author    = {Park, Hogun and Neville, Jennifer},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {3223--3230},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/447},
  url       = {https://doi.org/10.24963/ijcai.2019/447},
}


Misc
----

Datasets and Implementation of layers are following interfaces of Pytorch geometric.

