import json
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

SCOPES = ['https://www.googleapis.com/auth/calendar',
          'https://www.googleapis.com/auth/contacts.readonly',
          'https://www.googleapis.com/auth/contacts.other.readonly',
          'https://www.googleapis.com/auth/gmail.send',
          'https://www.googleapis.com/auth/gmail.readonly']


def parse_payload(payload: str) -> dict:
    try:
        json_payload = json.loads(payload)
        return json_payload
    except:
        return None

def initial_login():
    """Performs initial login and saves credentials."""
    flow = InstalledAppFlow.from_client_secrets_file(
        os.path.join(os.getcwd(), 'agent/secrets/client_secret_aura.json'), SCOPES)
    creds = flow.run_local_server(port=0)
    
    # Save the credentials for future use
    with open(os.path.join(os.getcwd(), 'agent/secrets/token.pickle'), 'wb') as token:
        pickle.dump(creds, token)
    
    return creds

def get_credentials():
    """Gets valid user credentials from storage."""
    creds = None
    if os.path.exists('agent/secrets/token.pickle'):
        with open('agent/secrets/token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = initial_login()
    
    return creds 

