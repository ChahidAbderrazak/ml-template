#!/bin/bash
eval "$($(which conda) 'shell.bash' 'hook')"
###-------------------------------------------#
clear && echo && echo " -> Activating env conda environment"
conda activate ./venv 

#|||||||||||||||||   VISUALIZATION    |||||||||||||||||
echo && echo " #################################################" 
echo " ##              VISUALIZATION                 ##" 
echo " #################################################" && echo 

uvicorn webapp:app --host 0.0.0.0 --port 8000

conda deactivate 