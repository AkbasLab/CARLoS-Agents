# CARLoS-Agents
CARLoS simulator with multiple agent implementations

# Setting up in Windows
pull repo

# Create new venv
py -3.11 -m venv venv
# Recommended to use python 3.11

venv\Scripts\activate

# Upgrade pip (make sure it is up to date)
pip install --upgrade pip

# Install Requirements
pip install -r requirements.txt

# Test the app
# 
python -m rlagent.test_env

# example here we are trying to train ppo
## TRAIN RL MODEL ##
python -m rlagent.train_ppo

## Run the model on CARLoS ##
python -m rlagent.test_ppo