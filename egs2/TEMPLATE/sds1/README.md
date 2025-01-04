# ESPnet-SDS

This is a template of **ESPnet-SDS** recipe.
ESPnet-SDS is an open-source toolkit for building unified web interfaces for various cascaded and end-to-end (E2E) spoken dialogue systems, supporting real-time automated evaluation metrics, and human-in-the-loop feedback collection.
This README describes features of ESPnet-SDS toolkit and provides comprehensive instructions for the users on navigating the demo interface.

---

## Features

### 1. Plug-and-Play Dialogue Systems
- Experiment with various **ASR**, **LLM**, and **TTS** systems by selecting from available options.
- Switch between:
  - **Cascaded Systems**: Separate ASR, LLM, and TTS components.
  - **End-to-End Systems**: Integrated processing pipelines.

### 2. On-the-Fly Evaluation Metrics
- Evaluate system performance using:
  - **Latency**: Response times.
  - **TTS Intelligibility**: Intelligibility of synthesized speech.
  - **TTS Speech Quality**: Clarity and quality of synthesized speech.
  - **ASR WER**: Word Error Rate for transcription accuracy.
  - **Text Dialogue Metrics**: Metrics like Perplexity and Diversity.

### 3. Human Feedback Collection
- Rate system responses using buttons for:
  - **Naturalness**: e.g., "Very Natural", "Somewhat Awkward".
  - **Relevance**: e.g., "Highly Relevant", "Slightly Irrelevant".
- Optional integration with a remote HuggingFace dataset as a backend database, allowing researchers to store human relevance judgments and log user interaction data, including input recordings and system outputs such as ASR transcripts, text responses, and audio responses.
  - **upload_to_hub** flag in `app.py` denotes the name of the remote HuggingFace dataset. If set to None, then this functionality is disabled.


---

## How to Use

1. **Start the Demo**
   - Run `run.sh` in `spoken_chatbot_arena/sds1` to run locally.
   - You can optionally also visit our [Voice Assistant Demo](https://huggingface.co/spaces/Siddhant/Voice_Assistant_Demo) on HuggingFace Spaces.
   - Wait for the interface to load.

2. **Choose a System Type**
   - **Cascaded System:**
      - Combines separate Automatic Speech Recognition (ASR), Language Model (LLM), and Text-to-Speech (TTS) systems.
   - **E2E System:**
      - An integrated model where all tasks are handled jointly.
   - Choose between **Cascaded** and **E2E** models.

3. **Configure the Components**
   - For a **Cascaded System**, configure the following:
     - **ASR:** Select the model for speech-to-text conversion.
     - **LLM:** Choose the language model for generating responses.
     - **TTS:** Select the model for converting text responses to speech.
   - For an **E2E System**, choose the integrated model

4. **Interact with the System**
   - **Input Your Voice:**
      - Click the **"Microphone"** button and start speaking.
      - The system will process your voice input and provide a synthesized speech response.

   - **View Outputs:**
      - **ASR Output:** Displays the text transcription of your speech.
      - **LLM Output:** Shows the generated text response.
      - **Audio Output:** Plays back the system's synthesized speech response.

5. **Evaluate Performance**
   -  Select an evaluation metric from **"Choose Evaluation Metrics"** to analyze specific aspects of the system:
   - Results will be displayed in the **"Evaluation Results"** box.

6. **Provide Feedback**
   - Use feedback buttons to rate the naturalness and relevance of the system's response.
      - **Naturalness:** Rate the quality of the synthesized speech.
      - **Relevance:** Rate how relevant the system's response was to your input.

---

## Requirements

### Browser Compatibility
- Use **Google Chrome** for the best experience.
- Web interface might not be compatible with certain web browsers like **Mozilla Firefox**

### Internet Connection
- A stable internet connection is required to access the demo.

---

## Troubleshooting

When running locally, please follow these suggestions to debug:

- **Gradio URL Issue**: If audio doesn't work on the Gradio public URL, use the **local URL** for better performance.
- **Docker Setup**: Ensure your environment's package versions match those in the [requirements file](https://huggingface.co/spaces/Siddhant/Voice_Assistant_Demo/blob/main/requirements.txt).
