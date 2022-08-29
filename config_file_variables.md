### model
- entry_point: Defines the file where the function for hyperparameter optimization is located

- function_name: The function that executes the hyperparameter optimization 
(training and evaluation of the model)


### data_options
- dataset_path: The path to a dataset. You cannot leave this variable empty. If you don't want to use a
dataset you still need to define a path to a dataset, even if you don't use it

- copy_data: If the chosen dataset should be copied into the workspace

### git_options
- git_uri: The git URI to the repository of your model

- branch: The branch that will be used for the hyperparameter search

### experiment
- use_local_workspace: If a local experiment folder should be created in root folder or a dedicated workspace 
directory (https://wiki.bwhpc.de/e/Workspace).


- cluster: Either "bwunicluster" or "horeka". Defines the cluster you are working on


- number_chain_jobs: How many times should a job - the suggestion evaluation - be chained together. 
It is used to cirumvent the problem of time outs in the cluster.
How to calculate the number of chained jobs:
  - t_eval: Time needed to run one evaluation cycle(Training and evaluating the model for one set 
  of hyperparameters)
  - parallel_bandwith: How many models will be evaluated in parallel, 
    this variable is defined in the section "sigopt_options"
  - time: The time a single job will run on the cluster. This variable is defined in the section "sigopt_options".
    The maximum time a job can run depends on the cluster. We suggest maximizing the time before chaining jobs.
  - observation_budget: How many suggestions will be created and evaluated. 
    This variable is defined in "sigopt_options" too.
  - number_chain_jobs = (observation_budget * t_eval) / (parallel_bandwith * time)
    Round this number up and maybe increase it a bit, to make sure, that the hyperparameter search has enough time.
    But if the time is not enough and the search stops before it is finished you still have the option to continue it.
  

- observation_budget: 60 # Max number of trials


- parallel_bandwidth: 4 # Number of parallel evaluations


- multimetric_experiment: Defines if multiple metrics/losses/scores should be evaluated. Default should be False.
If you set this to true your evaluation function needs to return a dict of the metrics:
```python
results = [{'name': 'name_of_metric_1',
            'value': value_of_metric_1},
           {'name': 'name_of_metric_2',
            'value': value_of_metric_2}]
return results
```

### parameters
In this paragraph we show parameter examples for some datatypes:
Make sure that the names used here match to the names you extract in the train and evaluation function 
(See Quickstart.md Section 2) 

#### Integer:
Range:
```
- name: max_depth
  type: int
  bounds:
    min: 1
    max: 1
```
Discrete Values:
```
- name: max_depth
  type: int
  grid:
    - 1
    - 2
    - 4
    - 8
```

#### Double:
Range:
```
- name: gamma
  type: double
  bounds:
    min: -5
    max: 5
```   
Discrete Values:
```
- name: gamma
  type: double
  grid:
    - -0.5
    - -0.1
    - 1.5
    - 4.3
```   

#### String/Categorical:
```
- name: criterion
  type: categorical
  categorical_values:
    - 'gini'
    - 'entropy'
``` 

#### Boolean:
We need to use a trick here, since there is no boolean option:
```
- name: use_xy
  type: int
  grid:
    - 0
    - 1
```
Option 2:
```
- name: use_xy
  type: categorical
  grid:
    - "False"
    - "True"
```
Using this might require changes in your code since the datatypes might not match.

#### None-Type:
We need to use a trick here, since there is no _None_ option:
```
- name: max_features
  type: categorical
  categorical_values:
    - 'sqrt'
    - 'log2'
    - 'None'
``` 
In this case you have to change your code, so that the String "None" gets converted to an actual _None_.

### metrics
In this section we give an example for single- and multi-metric optimization
Example for multi-metric optimization:
```yaml
metrics:
  - name: inception_score
    objective: maximize
    strategy: optimize
  - name: fid
    objective: minimize
    strategy: optimize
  - name: name_of_metric_3
    objective: minimize
    strategy: store
```
Example for a single-metric optimization:
```yaml
metrics:
  - name: inception_score
    objective: maximize
    strategy: optimize
```
Important: Remember to set the variable "multimetric_experiment" in section "experiment" accordingly.

- name: the name of the metric. Has to match one name in the dict returned by the evaluation function.
- objective: defines whether the metric should be maximized or minimized. The two possible values are 'maximize' and 'minimize'
  (without ' ')
- strategy: defines if SigOpt should optimize or only track the metric: Possible values: 'optimize' and 'store' (without ' ').

### sbatch_options
  - partition: Here you define the partition you want to use on the cluster. The names of possible partitions depend on the
cluster you use. 
    - The partitions with GPUs on the BWUniCluster:
      - "gpu_4" 
      - "dev_gpu_4"
      - "gpu_8"
    - The partitions with GPUs on the HoreKA cluster:
      - dev_accelerated
      - accelerated
  
  - gres: With this variable you can define resources you want to allocate on the server:
      - We recommend defining the number of GPUs to allocate:
        - 1 GPU: "gres: 'gpu:1' "
        - 2 GPUs: "gres: 'gpu:2' "
        - 3 GPUs: "gres: 'gpu:3' "
        - ...
        
        Without the " "
  - ntasks: Just set this to the "parallel_bandwith", see Section "sigopt_options"
  - mem-per-gpu: RAM allocated per GPU
  - time: The time a single job will run. For more information look at "number_chain_jobs" in section "experiment"


### sigopt_options TODO
  - dev_run: Used to toggle dev_run. If dev_run is true, the created suggestions are random and the hyperparameter search won't work.
    Use this option to debug your code and set it to false when your code works.


  - project_id: The id of your project. You can find this online when you are logged in to SigOpt


  - experiment_name: "tutorial_project" The name of the sigopt experiment. 
    Make sure that an experiment with this name exists or create one on SigOpt.


  - client_id: 11949