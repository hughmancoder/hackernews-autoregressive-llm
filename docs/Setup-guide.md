# Setup guide

### Setup Virtual Environment (if running locally)
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r .devcontainer/requirements.txt
```

*(Optional)* Install Taskfile CLI and zx for running task workflows locally:
```bash
brew install go-task
npm install -g zx
```

### Download Dataset
```bash
cd mainrun && python3 download_dataset.py && cd ..
```

### Running Docker Container

**Start the Docker Daemon**
```bash
open -a Docker
```

**Start the Dev Container**
1. Open the project folder in VS Code
2. Press `Cmd + Shift + P` 
3. Select **Dev Containers: Reopen in Container**

### Other commands

Refer to the `Makefile` for commands to train and test the model

## Files

`data.py`: Handles dataset loading,  Byte Pair Encoding (BPE) tokenizer training, and data batching.
`train.py`: Implements the training loop with hyperparameter optimisation.
`llama_model.py`: Defines the llama-style transformer model architecture.



