# Aura: Agent for Understanding, Reasoning and Automation

We develop a cascaded voice assistant system that includes ASR, TTS and a
ReAct based Agent for reasoning and action taking.

## Repository Structure

```
.
├── agent/                   # Core agent implementation
│   ├── actions/             # Action handlers for different tasks.
│   ├── controller/          # Agent state and control logic
│   ├── llm/                 # Language model integration
│   ├── secrets/             # Secure credential storage
│   └── agenthub/            # Agent implementations
│
├── ui/                      # User interface components
│   ├── local_speech_app.py  # Speech interface implementation (using gradio)
│   └── requirements.txt     # UI dependencies
|
└── environment.yaml        # Conda environment configuration
```

## Setup

1. Create the conda environment:
   ```bash
   conda env create -f environment.yaml
   ```

2. Set python Path
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```
3. Set LLM related environment variables:
    - `LLM_API_KEY`: API key for the language model
    - `LLM_API_BASE`: Base URL for the language model API
    - `LLM_MODEL`: Model identifier for the language model

4. Setup secrets (Required for tool use. Not needed if you are only using the chat functionality)
    Secrets are used to communicate with external APIs. follow the format in agent/secrets_example and change the name of the directory to agent/secrets
    You will need to setup a google Cloud Platform account, give necessary permissions, get the credential.json file. You will also need a SerpAPI key for web search.

5. Launch the gradio app:
 ```bash
    python ui/local_speech_app.py
 ```

 Adapted from https://github.com/Srijith-rkr/Aura
