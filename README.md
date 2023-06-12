<h1>HaPPy</h1> 

This is a graph neural network model for protein ligand binding affinity prediction, which implemented by PyG.

This work is published in AAAI-2023 conference and SCI journal "Interdisciplinary Sciences: Computational Life Sciences".

If you only want to know the main ideas and results of the work, please read this student abstract published on AAAI-2023, which is only two pages. But if you're interested in the full introduction, please read this journal article, which contains more details and experiments.

* Xianfeng Zhang, Yanhui Gu∗ , Guandong Xu, Yafei Li, Jinlan Wang and Zhenglu Yang. HaPPy: Harnessing the Wisdom from Multi-Perspective Graphs for Protein-Ligand Binding Affinity Prediction (Student Abstract)[C]. Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI 2023
* Xianfeng Zhang, Yanhui Gu∗ , Guandong Xu, Yafei Li, Jinlan Wang. A Multi-Perspective Model for Protein-Ligand Binding Affinity Prediction[J]. Interdisciplinary Sciences: Computational Life Sciences. To Appear.

<h2>Environment Installation</h2>

HaPPy requires Python 3.7+, PyTorch, PyG(PyTorch Geometric), openbabel, scipy, scikit-learn and some other common deep learning packages. 

If you're working on Windows, you can take advantage of the yaml file I've provided. However, if you want to experiment on Linux or other environments, please install the relevant packages by yourself.

<h2>Usage</h2>

<h3>Getting Raw Data</h3>

Please go to the official website of PDBbind dataset to download raw data([Welcome to PDBbind-CN database](http://pdbbind.org.cn/)), the 2016 version is used in this work. Then place the downloaded data in the project's "data" directory.

Due to the large size of the original data set, this project provides five samples as a demo.

<h3>Preprocessing Raw Data</h3>

Use the process.py file to process the raw data and convert the original 3D structure into a PyG format compliant 2D graph.

After processing, each data set becomes a PKL file.

Note that this file can only produce one dataset at a time, so you need to specify the training, validation, and test directories yourself.

<h3>Model Training</h3>

After setting the dataset directory, please directly use the train.py file for training, and you can modify the specific architecture and training hyperparameters of the model in this file. 

The training results are placed in the output directory, including the best model parameters, the performance of the model during training on the validation and test sets, and the final loss descent plot.
