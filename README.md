# CARLoS-Agents

### CARLoS simulator with multiple agent implementations

### Windows setup

step 1: clone the repo or download the zip file. if zip file extract the file locally.

step 2: open command prompt and cd to the project folder carlos_agents.

step 3: create a virtual environment by running command "python -m venv venv"

step 4: activate the venv by running "venv\Scripts\activate"

step 5: "pip install -r requirements.txt"

step 6: in the src folder > layouts> find the train_path.txt file and copy its file path.

step 7: in rlagent > carlos_gym_env.py > layout file_path variable update with the copied file path.

step 8: in command prompt run "python -m rlagent.[filename]"

### MacOS setup

step 1: After downloading the zip file, click on it to get the folder and put it on your desktop

step 2: Open your terminal and write "cd /Users/insert your name here/Desktop/CARLoS-Agents-main" and then press enter. If you want to copy it yourself, right-click on anything inside the main folder and copy the part that starts with 'Where:'

step 3: Write "source rl_env/Scripts/activate" in the terminal and press enter

step 4: Write "pip3 install -r requirements.txt" in the terminal and press enter

step 5: Write "python3 -m rlagent.test_env" in the terminal and press enter. If the terminal is on full screen, exit it because then you may not see the simulation

Now you should see logs in the terminal and a simulation window open.
