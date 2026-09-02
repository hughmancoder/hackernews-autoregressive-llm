# Notes

**Model**: A decoder-only autoregressive transformer with GPT-2 style architecture implemented from scratch in PyTorch 

**Task**: Causal next-token prediction. Given a sequence of previous tokens, the model learns to predict the next token in the sequence.

**Dataset**: ~100,000 Hacker News post titles downloaded from [Hugging Face](https://huggingface.co/datasets/julien040/hacker-news-posts).

[Training Log can be found here](Training%20Log.md)


## Local Environment Setup and Training 

Running natively on macOS leverages the Apple Silicon GPU via Metal Performance Shaders (mps), completing 7 epochs in ~3–5 minutes compared to ~40 minutes on CPU inside the Linux Docker container.

### 1. Setup Virtual Environment
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

### 2. Download Dataset
```bash
cd mainrun && python3 download_dataset.py && cd ..
```

## Running Docker Container

**Start the Docker Daemon**
```bash
open -a Docker
```

**Start the Dev Container**
1. Open the project folder in VS Code
2. Press `Cmd + Shift + P` 
3. Select **Dev Containers: Reopen in Container**


Refer to the `Makefile` for commands to train the model

# Sources

https://towardsdatascience.com/rope-clearly-explained/

