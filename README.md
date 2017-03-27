# Multi-process-UNREAL
Implementa UNREAL model by parameter-server architecture instead of the [multi-threading version](https://github.com/miyosuda/unreal).

## Version 0.1
* This version can make use all the cores of CPU rather than the original multi-threading version;
* INSTALLATION:
  1. install the [DeepMind Lab](https://github.com/deepmind/lab) environment
  2. copy this repository into the sub-directory "org_deepmind_lab" of any built project (say, the default random agent)
  3. run the training process by <code>python train.py</code>
* Tested with Ubuntu 16.04, anaconda2 and Tensorflow 1.0.1 
  
