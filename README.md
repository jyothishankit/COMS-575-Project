GENESIS is a background noise removal machine learning model based on CMGAN and MetricGAN<br>
It can remove any kind of background noise<br>
Detailed report:<br>
https://drive.google.com/file/d/182xXtN-PS_Fk1I55BTWytQ9M-ihrcxN-/view?usp=sharing<br><br>

Steps to run GENESIS on Nova CLuster:<br>
1) Install Python 3.9.16.<br>
https://www.python.org/downloads/release/python-3916/<br>
and then check PATH added to pick python version of 3.9.16 before other versions:<br>
->python --version<br>
Python 3.9.16<br>
2) Create virtual environment on top level of project.<br>
MacOS/Linux/Windows:<br>
python -m venv .<br><br>
3) Activate virutal env using below command:<br><br>
MacOS/Linux: <br>
source bin/activate<br>
Windows: Scripts/activate<br>
4) Check pip and python are from virtual env.<br>
which pip<br>
/virtual_env_path/bin/pip<br>
which python<br>
/virtual_env_path/bin/python<br>
5) Perform install of necessary packages.<br>
pip install -r requirements.txt<br>
6) Perform install of torch CUDA and torch audio CUDA wheels using below commands:<br>
pip install --extra-index-url https://download.pytorch.org/whl/cu113/ "torch==1.10.0+cu113"<br>
pip install --extra-index-url https://download.pytorch.org/whl/cu113/ "torchaudio==0.10.0+cu113"<br>
7) Make sure to check if GPU enabled devices can be found:<br>
>>python<br>
>>import torch<br>
>>torch.cuda.device_count()<br>
8) Copy contents of Dataset from the below link:<br>
VCTK Demand Datset UK: https://datashare.ed.ac.uk/handle/10283/2791 <br><br>

If needed to remove entire model cache:<br>
Delete contents of saved_model and run below command in interpreter:<br>
torch.cuda.empty_cache()<br>

9) Initiate job in SLURM cluster using below command:<br>
sbatch train575.sh<br>
(Run "scancel -u user_name" before this to kill running jobs)<br><br>
10) Change the contents of train575.sh to allow appropriate virtual env activation with name used.<br><br>
11) Samples of generated clean from noisy track can be found in <br>
Generated clean: https://github.com/jyothishankit/COMS-575-Project/tree/main/sample_clean<br>
Generated noisy: https://github.com/jyothishankit/COMS-575-Project/tree/main/sample_noisy<br>
