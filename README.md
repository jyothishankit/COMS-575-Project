Steps to run GENESIS on Nova CLuster:
1) Install Python 3.9.16.
https://www.python.org/downloads/release/python-3916/
and then check PATH added to pick python version of 3.9.16 before other versions:
->python --version
Python 3.9.16
2) Create virtual environment on top level of project.
MacOS/Linux/Windows: python -m venv .
3) Activate virutal env using below command:
MacOS/Linux: source bin/activate
Windows: Scripts/activate
4) Check pip and python are from virtual env.
which pip
/virtual_env_path/bin/pip
which python
/virtual_env_path/bin/python
5) Perform install of necessary packages.
pip install -r requirements.txt
6) Perform install of torch CUDA and torch audio CUDA wheels using below commands:
pip install --extra-index-url https://download.pytorch.org/whl/cu113/ "torch==1.10.0+cu113"
pip install --extra-index-url https://download.pytorch.org/whl/cu113/ "torchaudio==0.10.0+cu113"
7) Make sure to check if GPU enabled devices can be found:
>>python
>>import torch
>>torch.cuda.device_count()
8) Copy contents of Dataset from the below link:


If needed to remove entire model cache:
Delete contents of saved_model and run below command in interpreter:
torch.cuda.empty_cache()
9) Initiate job in SLURM cluster using below command:
sbatch train575.sh
(Run "scancel -u user_name" before this to kill running jobs)
10) Change the contents of train575.sh to 