# Project Documentation
[!] Install python 3.7 for Taxi's venv and 3.12 for Ollama's venv
- If you're on Mac, first run: 
```bash 
brew install swig.
```

Install from the wheel located in root with
```bash
 python -m pip install rlang/rlang/dist/rlang-0.2.5-py3-none-any.whl.
 ```
## Project Structure

This project contains two main components:

1. **Ollama** - Contains three sequential stages (stage1.py, stage2.py, stage3.py)
2. **Taxi** - Contains reinforcement learning implementations (dyna_q.py, q_learning.py, r_max.py)

```
.
├── .venv                  # Main virtual environment
│   ├── bin
│   ├── include
│   ├── lib
│   └── share
├── .gitignore
├── pip-selfcheck.json
├── pyvenv.cfg
├── ollama                 # Ollama component
│   ├── .ollama_venv       # Ollama-specific virtual environment
│   ├── stage1.py
│   ├── stage2.py
│   └── stage3.py
└── taxi                   # Taxi component
    ├── __pycache__
    ├── dyna_q.py
    ├── grounding.py
    ├── q_learning.py
    ├── r_max.py
    ├── taxi.rlang
    ├── vocab.json
    └── rlang-0.2.5-py3-none-any.whl
```

## Virtual Environments

This project uses multiple virtual environments:

1. **Main Environment** (`.venv/`) - Root-level virtual environment
2. **Ollama Environment** (`ollama/.ollama_venv/`) - Specific to the Ollama component

## Running the Ollama Component

To run the Ollama stages, you need to activate the Ollama-specific virtual environment:

```bash
# Navigate to the project root
cd path/to/project

# Activate the Ollama virtual environment
source ollama/.ollama_venv/bin/activate  # On Linux/Mac


# Run the stages in sequence
python3 ollama/stage1.py
python3 ollama/stage2.py
python3 ollama/stage3.py


```

## Running the Taxi Component

To run the Taxi reinforcement learning implementations, you need to activate the main virtual environment:

```bash
# Navigate to the project root
cd path/to/project

# Activate the main virtual environment
source .venv/bin/activate  # On Linux/Mac


python3 taxi/dyna_q.py
python3 taxi/q_learning.py
python3 taxi/r_max.py

```

## Reinforcement Learning Algorithms

The Taxi component contains three reinforcement learning algorithms:

1. **Dyna-Q** (`dyna_q.py`) - A model-based reinforcement learning algorithm that uses a model of the environment to simulate experience. Uses RLang policy with some probability and uses transition functions for initialization.

2. **Q-Learning** (`q_learning.py`) - A model-free reinforcement learning algorithm that learns the value of an action in a particular state. Utilizes Rlang reward functions to initialize the Q-table.

3. **R-Max** (`r_max.py`) - An algorithm that optimistically assumes maximum reward for unknown state-action pairs. Creates the empirical reward and transition tables using Rlang knowledge.

## Additional Files

- `taxi.rlang` - Likely a domain-specific language file for defining the taxi environment
- `vocab.json` - Vocabulary configuration
- `grounding.py` - Possibly handles grounding of symbols or concepts in the environment
- `rlang-0.2.5-py3-none-any.whl` - A wheel file for the rlang package (version 0.2.5)


