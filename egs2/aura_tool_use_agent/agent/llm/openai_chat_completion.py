from openai import OpenAI
import re
import re
import os

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = os.getenv("LLM_API_KEY", "EMPTY")
openai_api_base = os.getenv("LLM_API_BASE", "http://babel-1-1:8000/v1")
model = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

def get_response(messages,model=model):
    pass
    client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,)

    completion = client.chat.completions.create(model=model,messages=messages, temperature=0.1, max_tokens=2000)

    return completion.choices[0].message.content.strip()

def get_history_as_strings(history):
    output_string = ""

    for message in history:
        role = message["role"]
        content = message["content"]
        output_string += f"{role}: {content}\n"
    
    return output_string



def parse_simple_xml(text: str) -> dict:
    """
    Grab all <tag>value</tag> pairs from `text`, even if they sit inside an
    outer wrapper (or have junk before/after), and return a dict.

    • Strings '', 'none', 'null'  →  None
    • Strings 'true', 'false'     →  bool
    • If an <output> tag is present, add key 'has_output' (True/False)
    """
    # ── 1.  Find *all* simple tag–value pairs  ──────────────────────────────
    matches = re.findall(r"<(\w+)>\s*(.*?)\s*</\1>", text, re.DOTALL | re.IGNORECASE)

    result = {}
    for tag, raw in matches:
        val = raw.strip()

        if val.lower() in {"", "none", "null","None", "NULL"}:
            parsed = None
        else:
            parsed = val

        result[tag] = parsed

    return result


def parse_simple_xml(text: str) -> dict:
    """
    Grab all <tag>value</tag> pairs from `text`, even if they sit inside an
    outer wrapper (or have junk before/after), and return a dict.

    • Strings '', 'none', 'null'  →  None
    • Strings 'true', 'false'     →  bool
    • If an <output> tag is present, add key 'has_output' (True/False)
    """
    # ── 1.  Find *all* simple tag–value pairs  ──────────────────────────────
    matches = re.findall(r"<(\w+)>\s*(.*?)\s*</\1>", text, re.DOTALL | re.IGNORECASE)

    result = {}
    for tag, raw in matches:
        val = raw.strip()

        if val.lower() in {"", "none", "null","None", "NULL"}:
            parsed = None
        else:
            parsed = val

        result[tag] = parsed

    return result
