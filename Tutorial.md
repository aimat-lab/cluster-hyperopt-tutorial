# Tutorial

This Project is a Tutorial to set up your model for hyperparameter search using the 
[cluster-hyperopt repository](https://github.com/aimat-lab/cluster-hyperopt).

You can either create a new repository for this Tutorial and create all files etc.
as described in this document, or you can use this repository and read this Tutorial
as an explanation (you still need to create or change some files to get everything running).

This tutorial is separated into 8 Sections:

- Section 1: Creating a conda environment for this project

- Section 2: Creating a simple model, that is not yet usable for hyperparameter search

- Section 3: A function that will be called by the hyperparameter search algorithm

- Section 4: The config file, that defines all necessary parameters

- Section 5: Conda Environment 

- Section 6: SigOpt Tokens

- Section 7: Installing Cluster Hyperopt on the BWUniCluster

- Section 8: Running the hyperparameter search

### Section 1
In this section we will create the conda environment needed for our model that we develop.
It is used to install all dependencies that our model will need.
In this tutorial we will call this environment 'model_env'.


Open a console on your local machine.
We assume that you have already installed conda.
Create a new environment using ```conda create -n "model_env" ```. 
You can name the environment how you want. We chose "model_env".
Activate the environment using ```conda activate model_env```.
Make sure pip is installed.
Install sklearn using pip:```pip install -U scikit-learn```

### Section 2
In this section we will create a simple model.

We create a file ```model.py```.
It contains the class ```Model```, 
which has functions to create, train and evaluate a random forest classifier imported from sklearn.

As you can see, the train function allows us to set the number of trees (```n_estimators```), 
the splitting criterion (```criterion```), the maximum depth for a tree (```max_depth```), 
if we want to use bootstrapping (```bootstrap```) and how many random features 
should be considered when calculating a split (```max_features```).
```py
from sklearn.ensemble import RandomForestClassifier


class Model:
    rfc = None      # The Random Forest Classifier

    random_state = None

    def __init__(self, random_state=0):
        self.random_state = random_state

    def train(self, X_train, y_train, n_estimators, criterion, max_depth, bootstrap, max_features):
        self.rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                          bootstrap=bootstrap, max_features=max_features,
                                          random_state=self.random_state)
        self.rfc.fit(X=X_train, y=y_train)

    def evaluate(self, X_test, y_test):
        return self.rfc.score(X_test, y_test)
```

We now need a function that loads a dataset, creates a Model, trains and evaluates it. 
Therefore we create the file ```main.py``` with the function ```run_without_hyperopt()```.
This function trains the model with the following hyperparameters:
   - n_estimators = 5
   - criterion = 'entropy'
   - max_depth = 5
   - bootstrap = True
   - max_features = 'sqrt'

Now we can execute this function and will see the accuracy of the model in the console.
```py
from model import Model
from sklearn import datasets
from sklearn.model_selection import train_test_split

random_state = 0


def run_without_hyperopt():
    # load and split dataset
    ds = datasets.load_wine()
    X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=0.33, random_state=random_state)

    # train and evaluate model
    model = Model()
    model.train(X_train=X_train, y_train=y_train, n_estimators=5, criterion='entropy', max_depth=5,
                bootstrap=True, max_features='sqrt')
    print(model.evaluate(X_test=X_test, y_test=y_test))


if __name__ == '__main__':
    run_without_hyperopt()
```


### Section 3
In this section we create a function that will be called by cluster_hyperopt
and returns the accuracy and metadata of the model. It is the only function in our models repository that
cluster_hyperopt will execute.
We add this function to the ```main.py``` file.
```run_hyperopt(hyperopt_config=None)``` takes a run configuration as input. 
This configuration contains the suggested hyperparameters for our model.
   1. In part 01 the function extracts the suggestion from the hyperopt_config
   and from the suggestion it can get the values for each hyperparameter
   2. Part 02 is about loading the dataset and splitting it into the train and test parts.
   3. In part 03 the function trains and evaluates the model. Afterwards it returns 
   the result (accuracy) and the metadata. 
   This is important, since the optimizer expects this return format.
```py
def run_hyperopt(hyperopt_config=None):
    # Extract options (Part 01)
    suggestion = hyperopt_config['suggestion']

    n_estimators = suggestion['n_estimators']
    criterion = suggestion['criterion']
    max_depth = suggestion['max_depth']
    bootstrap = suggestion['bootstrap']
    max_features = suggestion['max_features']

    # load and split dataset (Part 02)
    # this is not optimal, since we only need to download the dataset once. We only use this for simplicity
    ds = datasets.load_wine()
    X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=0.33, random_state=random_state)

    # train and evaluate model (Part 03)
    model = Model()
    model.train(X_train=X_train, y_train=y_train, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                bootstrap=bootstrap, max_features=max_features)

    result = model.evaluate(X_test=X_test, y_test=y_test)
    metadata = None
    print(result)

    return result, metadata
```
Right now it is not yet possible to start the hyperparameter search. We first need to write a 
configuration file, which defines all parameters needed for the hyperparameter search.


### Section 4
In the 4. Section we create the configuration file needed to run the hyperparameter search.
You can save this file at an arbitrary location on the BWUniCluster. 
But memorize the path to it, since you will need it again in Section 8.

Do not confuse this configuration file with the hyperopt_config parameter
for the ```run_hyperopt(hyperopt_config=None)``` function, created in section 3.
The config file can be split in roughly 8 parts:
   - **model**
   - **git_options**
   - **data_options**: Not used here, but might be important: 
       Defines options for the dataset, that might be used
   - **experiment**
   - **parameters**: defines all hyperparameters
   - **metrics**: defines the kind of metric and whether to minimize or maximize it.
   - **sbatch_options**: Contains the sbatch options (see Slurm), to add to the execution script. 
                     Important Note: What happens with those options is different for HoreKA 
                     and BWUniCluster, since the parallelization of both differs.
   - **sigopt_options**

We will create the file step by step. 

A complete config file can be found in this repository. See: [config.yaml](https://github.com/u-adrian/Tutorial/blob/main/config.yaml)

Starting with the model options:
```yaml
model:
    entry_point: "main.py" # The python file name that includes the function for evaluating the suggestions
    function_name: "run_hyperopt"
```

- ```entry_point``` is the python file that includes the ```run_hyperopt(hyperopt_config=None)``` function
- ```function_name``` defines the name of the function that will be executed by the hyperparameter search algorithm.
    In our case: ```"run_hyperopt"```
--------------------------------------------------------------------
The next part is about the git options:
```yaml
git_options:
    git_uri: "git@github.com:u-adrian/Tutorial.git"
    branch: "main" # Either branch or version can be used. Using the option version allows to load specific tags
```
- ```git_uri```: The uri to the model repository. This is "git@github.com:u-adrian/Tutorial.git" for this project.
- ```branch``` defines the branch of the repository.
--------------------------------------------------------------------
```yaml
experiment:
    cluster: "bwunicluster"  # Either "bwunicluster" or "horeka"
    number_chain_jobs: 4 # How many times should a job - the suggestion evaluation - be chained together. It is used to
                       # cirumvent the problem of time outs in the cluster
    parallel_bandwidth: 4
    observation_budget: 60
```

- ```cluster``` Defines the algorithm to use. You can choose between "horeka" and "bwunicluster".
    This setting has a great effect what happens with the "sbatch_options".
- ```number_chain_jobs``` Since there is a maximum time a job can run on the clusters we need to chain jobs together.
- ```parallel_bandwidth``` is the number of suggestion that can be created simultaneously. Set this to the
  number of models you evaluate in parallel.
- ```observation_budget``` defines how many hyperparameter-suggestions can be created in total.
--------------------------------------------------------------------
Now we define the hyperparameters that we want to optimize.
- The first hyperparameter is ```max_depth```. The datatype must be integer and we restrict it to an interval from 
    1 to 10.
- We chose the same settings for the number of 
    trees (```n_estimators```) in the forest. 
- The third parameter ```bootstrap``` is a boolean. 
    Since it is not possible to define booleans in "SigOpt" we use the integers 0 and 1.
    We use "grid" in this example, so you can see how to define a set of integers.
- ```max_features``` is a categorical value, and we chose the two options "sqrt" and "log2".
- ```criterion``` is also of type "categorical" with the possible values "gini" and "entropy".
```yaml
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
```
---------------------------------------------------------------------
Now we define the metrics we want to optimize:
The objective is to maximize it. Set the strategy to "optimize"
```yaml
metrics:
  - name: accuracy
    objective: maximize
    strategy: optimize
```
----------------------------------------------------------------------
The ```sbatch_options``` are a crucial part to get the hyperparameter search run properly on
the BWUniCluster and HoreKA. Since the choice of the cluster affects those settings we will discuss the settings
for HoreKA and also for the BWUniCluster. You can find a brief description of both implementations here TODO.

- We use the partition ```dev_gpu_4``` on the horeka cluster.

Usage of other variables is possible but only optional. Examples can be found 
[here](https://github.com/u-adrian/Tutorial/blob/main/config_file_variables.md#sbatch_options)

[Here](https://slurm.schedmd.com/sbatch.html) you can find a description for all sbatch options.

```yaml
sbatch_options:
  partition: "dev_gpu_4"
```
-------------------------------------------------------------------------
In the last step we define the ```sigopt_options```.
- ```dev_run: true``` is used to test your implementation. 
    The generated suggestion cannot be used to find good hyperparameters.
    After you tested your application with ```dev_run: true``` and everything works you can 
    change it to ```dev_run: false```, with this setting SigOpt will create useful suggestion.
- ```project_id``` is the ID of your Project, that you created on SigOpt. Make sure, that a project with 
                this ID already exists.
- ```experiment_name```: the name of the experiment. Make sure, that no experiment with this name exists in the project with id ```project_id```
- ```client_id``` Is your Identifier for SigOpt. 
  To find your client id, you need to login to SigOpt and open "API Tokens".

```yaml
sigopt_options:
  dev_run: false # If the dev api of sigopt should be used to get the suggestions
  project_id: "test_project"
  experiment_name: "tutorial_project"
  client_id: 11949
```
--------------------------------------------
A complete config file can be found here: [config.yaml](https://github.com/u-adrian/Tutorial/blob/main/config.yaml)

### Section 5
To let our hyperparameter search run on the server we need a conda environment, 
which includes all dependencies for our model. Right now this environment only exists on our local machine.
We created it in section 2 and named it "model_env"

We have two options now:
1. Create an environment by hand on the server and add its name in the config file:
   ```yaml
   experiment:
      cluster: "bwunicluster"  # Either "bwunicluster" or "horeka"
      number_chain_jobs: 4 # How many times should a job - the suggestion evaluation - be chained together. It is used to
                       # cirumvent the problem of time outs in the cluster
      parallel_bandwidth: 4
      observation_budget: 60
      conda_env: name_of_env 
   ```
   This code snippet shows an updated version of the experiment configuration variables. 
   Note that we added a new variable at the end "conda_env". It holds the name of the conda env you
   want to use.
-----
2. Create an ```environmenmt.yaml``` file and add it to the topmost folder of your repository as it is 
   done in this project. At this location cluster_hyperopt can find it.
   The environment.yaml file contains all information about an environment, 
   so that cluster hyperopt can create a conda environment from it on the server.

   To create such file open a terminal and activate the ```model_env``` using:

   ```conda activate model_env```

    In the next step create the ```environment.yaml``` file:

    ```conda env export > environment.yaml```
    
    Move this file to the correct location, described above.

### Section 6
To have access sigopt from the server you need the API Token and Development Token.
We create a file ```sigopt_token``` where we store those tokens. You can create this file 
in a location of your choice on the BWUniCluster:

```
SIGOPT_TOKEN=**********************************************
SIGOPT_DEV_TOKEN=**********************************************
```

You can find your tokens on the SigOpt website: _Your Username_ >
Then we store the path to this file in a system variable:

```export SIGOPT_ENV_FILE=path/sigopt_token```

Now cluster hyperopt can connect to sigopt.

### Section 7

One last thing we need to do before using cluster_hyperopt is to install it:
Open a terminal in a location of your choice on the BWUniCluster.
Use an existing or create a new conda environment.

```conda env create -n "a_new_env"```

And activate it:

```conda activate a_new_env```

Clone the cluster_hyperopt repository

```git clone git@github.com:aimat-lab/cluster-hyperopt.git```

And install it (make sure that pip is installed):
```
cd cluster-hyperopt
pip install -e .
```

### Section 8

Now we can run the hyperparameter search.

Open a terminal on the BWUniCluster and execute the following command

```python .../sigopt_hyperopt/hyperopt.py start --config_path=.../config.yaml```

/sigopt_hyperopt/hyperopt.py is located in the cluster-hyperopt repository, that you installed in Section 7.

--config_path is the path to the configuration file you created in Section 4.