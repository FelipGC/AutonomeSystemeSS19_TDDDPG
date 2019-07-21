# 3D Ball Balance
Autonome Systeme SS2019
___

- Set path at: `env = UnityEnvironment(file_name=<PATH>)` in run.py
<br /> If you want to start training directly with Unity set `env = None
<br />(default path =  `"buildings/Linux/3DBall_train_linux.x86_64"`)

- Start a training session with `run.py`

- Start the Multiple-Hyperparameter training sessions with `hyperparameter_training.py`

- All hyperparameters with corresponding values can be found in `params.py`

- For starting the dashboard just run `dashboard.py`
<br/> The dashboard will take the logs from the directory `/logs_dash`. 
<br/>If you want to use your own generated logs after training change the path `PATH_DASH = 'logs_dash'`to `PATH_DASH = 'logs'` in `params.py`

- To use generated the weights after a training change `TRAINING_MODE = True` to `TRAINING_MODE = False` in `params.py`
___


### Hyperparameters

| Hyperparameter| Values| Description  |
| ------------- |:-------------:| -----:|
|**POLICY_DELAY**| 1, 2, 4, 8, 16 | _update global networks  and the policy every n steps_ |
| **TAU**| 1, 0.1, 0.01, 0.001, 0.0001, 0.00001| _soft update_ |
| **BUFFER_SIZE**| 8192, 262144, 1048576| _replay memory size_ |
| **UNITS**| 64, 128| _-_ |
| **BATCH_SIZE**| 32, 64| _minibatch size_ |
| **INITIAL_RANDOM_ROLLOUTS**| True, False| _True if we want to start with completely random, actions_ |
| **DROP_OUT_PROB**| 0, 0.1, 0.2| _-_ |


