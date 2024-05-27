#!/bin/bash
eval "$($(which conda) 'shell.bash' 'hook')"
python_version=3.8

####################################################
clear && echo && echo " -> Setup env conda environment"
conda create -p ./venv python=$python_version conda -y

echo && echo " -> Activating conda environment"
conda activate ./venv
 
echo && echo " -> Install pip packages"
pip install -r requirements.txt 
pip install PyQt5==5.15.10 PyQt5_sip>=12.8.1 opencv_python>=4.1.2.30
conda deactivate 
