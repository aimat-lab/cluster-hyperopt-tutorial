# Quickstart

This file will enable you to set up your model for hyperparameter search using the 
[cluster-hyperopt repository](https://github.com/aimat-lab/cluster-hyperopt).

It is separated into 3 Sections:

- Section 1: Preconditions

- Section 2: Creating a wrapper function, which executes the training and evaluation of your model

- Section 3: Installation of cluster-hyperopt and creation of necessary files


A more detailed Tutorial can be found in the [Tutorial.md](https://github.com/u-adrian/Tutorial/blob/main/Tutorial.md) file.

### Section 1 Preconditions
You need:
- access to SigOpt
- access to BWUniCluster or HoreKA
- a GitHub repo for your project/model
- access to Repositories: make sure you have access to [cluster-hyperopt](https://github.com/aimat-lab/cluster-hyperopt)
and your models repository from the server. Create an ssh key pair if needed.
- a conda environment (we will call it "E1") for your model on the BWUniCluster/HoreKA
- a training and an evaluation function for your model


### Section 2
In this section we make your model compatible with Cluster Hyperopt.
Therefore, we create a wrapper function (lets call it "run_hyperopt(config)", but you can name it how you want), 
that trains and evaluates the model. This is the only function that Cluster Hyperopt will execute from your model repository.
  - Input: python dict
  - Output: The loss/score/metric of the evaluated model
  - Structure:
    - extract hyperparams from the input dict
    - run the training algorithm of your model using the extracted hyperparameters
    - run the evaluation algorithm of your model
    - return the loss/score/metric of your model
  - Location: In an arbitrary python file in your github repo. Make sure, that it is uploaded to github.


Example structure of "run_hyperopt(config)":
```python
    def run_hyperopt(config):
        # extract hyperparams. Example:
        max_depth = config["suggestion"]["max_depth"]
        
        #run the training algorithm. Example:
        model = train(max_depth)
        
        #evaluiate the model. Example:
        loss = evaluate(model)
        
        #return loss/score/metric
        return loss

```


The python dict mentioned as input has the following structure:
We added some example hyperparameters.
```
dict: {
    'suggestion': {
        "hyperparameter_x": value    # Example Hyperparameter
        "bootstrap": 1,              # Example Hyperparameter
        "criterion": "gini",         # Example Hyperparameter
        "max_depth": 5,              # Example Hyperparameter
        "max_features": "log2",      # Example Hyperparameter
        "n_estimators": 2            # Example Hyperparameter
    },
    'dataset': 'path/to/dataset/'
    'output_path': path/to/output/dir
}    
```

### Section 3
#### Installation of cluster-hyperopt
  - download the repository [cluster-hyperopt](https://github.com/aimat-lab/cluster-hyperopt) to a
  location of your choice on the BWUniCluster or HoreKA.
  - you can use an existing conda environment or a new one to install cluster-hyperopt
  (we will call it "E2")
  - open a shell in the topmost directory of the cluster-hyperopt repository
  - activate the conda environment of your choice (E2) and execute "pip install -e ." 
  This will install cluster-hyperopt into the active environment.
  

#### Creation of the config file 
The config.yaml file is needed so that cluster_hyperopt can find all the
necessary information for a hyperparameter search.
  - You can create it in an arbitrary place on the server (BWUniCluster/HoreKA)
  - See [config_file_variables.md](https://github.com/u-adrian/Tutorial/blob/main/config_file_variables.md) for information about every variable
  - Parameters and metrics need to match those used in [Section 2](https://github.com/u-adrian/Tutorial#Section-2) 
  - Here is a minimalistic configuration file that you can fill out and use:
    ```yaml
    model:
      entry_point: TODO # The python file name that includes the function for evaluating the suggestions
      function_name: TODO # the function that executes the training and evaluation
    data_options:
      dataset_path: TODO # can be empty eg: ""
    git_options:
      git_uri: TODO
      branch: TODO # Either branch or version can be used. Using the option version allows to load specific tags
    experiment:
      cluster: TODO  # Either "bwunicluster" or "horeka"
      number_chain_jobs: TODO # How many times should a job - the suggestion evaluation - be chained together. It is used to
                           # cirumvent the problem of time outs in the cluster
      observation_budget: TODO # Max number of trials
      parallel_bandwidth: TODO # Number of parallel evaluations
      conda_env: TODO # name of the conda env used for the model. This is E1, see "Section 1: Preconditions"
    parameters:
      TODO
    metrics:
      TODO
    sbatch_options:
      partition: TODO
    sigopt_options:
      dev_run: true # Change this to false if you debugged your program
      project_id: TODO
      experiment_name: TODO
      client_id: TODO
    ```
  
  - Example file with some example values:
    ```yaml
    model:
      entry_point: "main.py" # The python file name that includes the function for evaluating the suggestions
      function_name: "run_hyperopt" # the function that executes the training and evaluation
    data_options:
      dataset_path: "/path/to/dataset" # can be empty 
    git_options:
      git_uri: "git@github.com:u-adrian/Tutorial.git"
      branch: "main" # Either branch or version can be used. Using the option version allows to load specific tags
    experiment:
      cluster: "horeka"  # Either "bwunicluster" or "horeka"
      number_chain_jobs: 4 # How many times should a job - the suggestion evaluation - be chained together. It is used to
                           # cirumvent the problem of time outs in the cluster
      observation_budget: 60 # Max number of trials
      parallel_bandwidth: 4 # Number of parallel evaluations
      multimetric_experiment: false
      conda_env: name_of_env # name of the conda env used for the model 
    parameters:
      - name: max_depth
        type: int
        bounds:
          min: 1
          max: 10
      - name: n_estimators
        type: int
        bounds:
          min: 1
          max: 10
      - name: bootstrap
        type: int
        grid:
          - 0
          - 1
      - name: max_features
        type: categorical
        categorical_values:
          - 'sqrt'
          - 'log2'
      - name: criterion
        type: categorical
        categorical_values:
          - 'gini'
          - 'entropy'
    metrics:
      - name: accuracy
        objective: maximize
        strategy: optimize
    sbatch_options:
      partition: "dev_gpu_4"
    sigopt_options:
      dev_run: false # If the dev api of sigopt should be used to get the suggestions
      project_id: "test_project"
      experiment_name: "tutorial_project"
      client_id: 11949
    ```
 
 
#### Creation of SigOpt Token file:
 
To get value suggestions for the hyperparameters from SIGOPT, cluster_hyperopt needs the API Tokens.
We create a file ```sigopt_token```(but you can name it how you want) where we store those tokens. 

You can create this file 
in a location of your choice on the BWUniCluster/HoreKA:

```
SIGOPT_TOKEN=**********************************************
SIGOPT_DEV_TOKEN=**********************************************
```
The difference between these two tokens is that the SIGOPT_DEV_TOKEN only provides
a random suggestion for developing purposes whereas SIGOPT_TOKEN provides serious
suggestions. This option can then be activated or deactivated in the config file via 
sigopt_options.dev_run

You can find your SIGOPT_TOKEN and SIGOPT_DEV_TOKEN on the 
SIGOPT Website: https://app.sigopt.com/tokens/info under the menu points: 
- \<Your Username\>
  - "API Tokens"

The next step is to assign an environment variable called SIGOPT_ENV_FILE 
to the path of this newly created file:

```export SIGOPT_ENV_FILE=path/sigopt_token```

Now cluster hyperopt can find and access the file, read out the tokens and 
request value suggestions from SIGOPT. 
### Section 4 Run HyperOpt
- Open shell in arbitrary folder (but on the cluster).
- activate the conda env where cluster-hyperopt is installed (E2)
  ```bash
  conda activate name_of_E2
  ```
- execute cluster hyperopt:
  ```bash
  python /.../cluster-hyperopt/sigopt_hyperopt/hyperopt.py start --config_path=/path_to_config/config.yaml
  ```
  The hyperopt.py file which will be executed is located in the cluster_hyperopt repository that
  you downloaded and installed at the beginning of Section 3: [Installation of cluster-hyperopt](https://github.com/u-adrian/Tutorial#Installation-of-cluster-hyperopt).
