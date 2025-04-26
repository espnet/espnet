import requests
from agent.actions.action import Action
from agent.controller.state import State
import json

class WebSearchAction(Action):
    def __init__(self, thought: str, payload: str):
        super().__init__(thought, payload)
        with open("agent/secrets/secrets.json", "r") as f:
            self.serpai_api_key = json.load(f)["serpai_api_key"]
        self.engine="google"

    def clean_text(self, text: str) -> str:
        return text.encode('ascii', 'ignore').decode()
        
    def execute(self, state: State) -> str:
        query = self.payload
        
        params = {
            "engine": self.engine,
            "q": query,
            "api_key": self.serpai_api_key,
            "num":3
        }

        try:
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
        except Exception as e:
            return f"Error: {e}"

        if response.status_code != 200:
            return "Error: Failed to fetch search results"
        
        data = response.json()
        results = data.get("organic_results", [])

        if not results:
            return "No search results found"
        
        results_filtered = [{'snippet': result['snippet'], 'snippet_highlighted_words': result['snippet_highlighted_words'], 'title': result['title'], 'source': result['source']} for result in results]
        
        state.history.append({
            'action': {'type': 'web_search', 'payload': self.payload},
            'observation': {"type": "web_search", "payload": self.clean_text(json.dumps(results_filtered))}
        })

        return json.dumps(results_filtered)
    
if __name__ == "__main__":
    action = WebSearchAction(thought="", payload="weather in Pittsburgh today")
    print(action.execute(State()))
        
        