from agent.actions.action import Action
from agent.controller.state import State
from googleapiclient.discovery import build
from agent.actions.utils import get_credentials
import json

class ContactAction(Action):
    def __init__(self, thought: str, payload: str):
        super().__init__(thought, payload)
        
    
    def execute(self, state: State) -> str:
        creds = get_credentials()
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me',maxResults=7).execute()
        contacts =[]
        if 'messages' in results:
            for msg in results['messages']:
                msg_id = msg['id']
                msg = service.users().messages().get(userId='me', id=msg_id, format='metadata').execute()
                for header in msg['payload']['headers']:
                    if header['name'] == 'From' or header['name'] == 'To':
                        contacts.append(header['value'])
        state.history.append({"action": {"type": "contact", "payload": None}, "observation": {"type": "contact", "payload": json.dumps({"length": len(contacts), "contacts": contacts})}})
        return json.dumps({"length": len(contacts), "contacts": contacts})
    
if __name__ == "__main__":
    action = ContactAction("I need to send an email to John Doe", None)
    print(action.execute(State()))

    

    # Search inbox for a known name
    
        
        