# Quickstart

This file is made to enable you to set up your model for hyperparameter search using the 
[cluster-hyperopt repository](https://github.com/aimat-lab/cluster-hyperopt) .

This Quickstart is separated into X Sections:

- Section 1: Preconditions

- Section 2: Creating a wrapper function, which executes the training and evaluation of your model

- Section 3: Installation of cluster-hyperopt and creation of necessary files

### Section 1: Preconditions
- a conda environment for your model
- a training and an evaluation function for your model
- access to SigOpt
- access to BWUniCluster or HoreKA
- a GitHub repo for your project/model


### Section 2:
create a wrapper function (lets call it "run_hyperopt", but you can name it how you want), 
that trains and evaluates the model.
  - Input, output:
    - Input: python dict
    - Output: The loss/score/metric of the evaluated model
  - Structure
    - extract hyperparams from the input dict
    - run the training algorithm of your model
    - run the evaluation algorithm of your model
    - return the loss/score/metric of your model
  - Location: In an arbitrary python file in your github repo. Make sure, that it is uploaded to github.

### Section 3:
- Installation of cluster-hyperopt:
  - download the repository [cluster-hyperopt](https://github.com/aimat-lab/cluster-hyperopt) to a
  location of your choice on the BWUniCluster or HoreKA.
  - you can use an existing conda environment or a new one to install cluster-hyyperopt
  - open a shell in the topmost directory of your local cluster-hyperopt repository
  - activate the conda environment of your choice and execute "pip install -e ."



- Since you need an environment for your model: The environment.yaml file:
  - Is needed to create a conda environment with all dependencies your model needs
  - needs to be in the topmost folder of your projects repo.
  - can be created by opening a shell, activating the model env. and executing 
  ```conda env export > environment.yaml```. (Note that the environment.yaml will be
  created in the directory, in which you created the shell. You might need to move it)
  
- Creation of the config file:
  - arbitrary place on the server
- SigOpt Token file
  - content of the file
  - name of the file: arbitrary
  - arbitrary location on the bwunicluster or horeka
  - How to make it accessible to cluster-hyperopt: ```export SIGOPT_ENV_FILE=path_to_this_file```

### Section 4: Run HyperOpt
- Open shell in arbitrary folder (but on the cluster).
- activate the env, where cluster-hyperopt is installed
-  .........
