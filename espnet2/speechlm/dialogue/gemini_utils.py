#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Generate and process SpeechLM SFT data using Gemini-APIs

Before you run:
(1) install Google Cloud Client. You can install by conda:
    conda install -c conda-forge google-cloud-sdk
(2) install vertexai:
    pip install vertexai
(3) login with google cloud account running:
    gcloud init
(4) create the credential file:
    gcloud auth application-default login --project PROJECT_ID
(5) Enable this Gemini service at the web:
    https://console.developers.google.com/apis/api/aiplatform.googleapis.com/overview?project=slm-sft
"""

import logging
import json

from espnet2.speechlm.dialogue.dialogue_format import Dialogue

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
except:
    vertexai = None
    GenerativeModel = None
    GenerationConfig = None

##### MACROS #####
PROJECT_ID = "lti-sw-gemini" # revise this to your account

##### Section 1: Prompt #####
s2s_prompt = """
You will be provided with a text-based conversation between an AI Assistant and a user. Sometimes, there is a system prompt to set the context.
Your task is to convert this text-based conversation into a speech-based conversation, and output the transcription in JSON format.

Here are some detailed instructions:
(1) Your response should be in JSON format. The response should be a List of Dict; each dict should contain the role and content. The role can only be system, user or assistant.
(2) Your response should be in English only.
(3) Your response should be transcriptions that are conversational and can be easily pronounced. Specifically, no linebreak and no special symbols. Convert digit numbers into words.
(4) Based on the text conversation, the speech-based conversation can have single or multiple turns.
(5) The total length of the conversation should be no longer than 200 words. Each turn of the conversation should be no longer than 70 words.
(6) If there is a system prompt, try to keep it in text format. However, if that system prompt is not suitable for speech-based conversation, try to revise it accordingly.

Text-based conversation:
"""

prompt_dict = {
    "s2s": s2s_prompt,
}

##### Section 2: LLM setups, configurations and processing functions #####
class GeminiAPIInterface:
    def __init__(
        self,
        model_id: str,
        prompt_method: str = None,
    ):
        
        assert vertexai is not None, "Please install vertexai"
        vertexai.init(project=PROJECT_ID)
        
        self.model_id = model_id
        self.prompt_method = prompt_method

        self.set_prompt_method(model_id, prompt_method)
    
    def set_prompt_method(self, model_id, task = None):
        if task is not None:
            instruction = prompt_dict[task]
        else:
            instruction = None

        logging.info(f"Switch to model {model_id} with task {task}")
        
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            max_output_tokens=1024,
        )
        self.model = GenerativeModel(
            model_id, 
            system_instruction=instruction,
            generation_config=generation_config,
        )

        self.task = task
        self.model_id = model_id
        
    def generate_spoken_dialogue(
        self,
        name,
        dialogue, 
        model_id,
        task: str = "audio_dialogue",
        max_len_words: int = 500,
    ):
        if task not in ["audio_dialogue", "audio_text_dialogue"]:
            raise ValueError(f"invalid task: {task}")

        if task != self.task or model_id != self.model_id:
            self.set_prompt_method(model_id, task)

        # (1) check if the string is too long
        dialogue_str = dialogue.to_str()
        if len(dialogue_str.split()) > max_len_words:
            logging.warning(f"Dialogue {name} is discarded as it's too long ")
            return None
        
        # (2) check if the string contains code
        if "```" in dialogue_str:
            logging.warning(f"Dialogue {name} is discarded as it contains code")
            return None
        
        # (2) model call
        response = self.model.generate_content(dialogue_str)

        # (3) parse into json format
        try:
            response = response.text.removeprefix("```json").removesuffix("```")
            response = json.loads(response)
        except:
            logging.warning(
                f"Dialogue {name} failed to be parsed into json dict. Skip. \n"
                f"Returned dict: {response}"
            )
            return None
        
        # (4) sanity check
        if isinstance(response, dict) and len(response) == 1:
            response = next(iter(response.values()))

        if not isinstance(response, list):
            logging.warning(f"Dialogue {name} not a list {response}")
            return None
        
        dialogue = Dialogue("text_dialogue")
        for idx, d in enumerate(response):
            if not ("role" in d and "content" in d):
                logging.warning(f"Dialogue {name} | segment {idx} incomplete")
                return None
            
            role, content = d['role'], d['content']
            if role not in ["system", "user", "assistant"]:
                logging.warning(f"Dialogue {name} | segment {idx} invalid role {role}")
                return None
            
            dialogue.add_segment(role, "text_bpe", content)
        
        return dialogue