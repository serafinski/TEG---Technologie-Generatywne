import yaml
from pathlib import Path

class PromptManager:
    def __init__(self):
        self.prompts = self._load_prompts()
        
    def _load_prompts(self):
        prompts = {}
        prompt_dir = Path(__file__).parent.parent / "prompts"
        
        for yaml_file in prompt_dir.glob("*.yaml"):
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                print("data:", data)
                agent_type = yaml_file.stem
                prompts[agent_type] = {
                    'description': data.get('description', f"Agent for {agent_type}"),
                    'template': data.get('template', "")
                }
        return prompts


    def get_template(self, agent_type):
        return self.prompts.get(agent_type, {}).get('template', "")
    
    def get_description(self, agent_type):
        return self.prompts.get(agent_type, {}).get('description', "")
