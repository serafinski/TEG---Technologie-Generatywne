import yaml
from dotenv import load_dotenv
import os

class Config:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Load values from environment variables
        self.key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", 0))

        # Load additional configurations from config.yml
        self._load_yaml_config()

    def _load_yaml_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yml")
        try:
            with open(config_path, 'r') as file:
                yaml_config = yaml.safe_load(file)
                self.yaml_config = yaml_config  # Store YAML data as a dictionary
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_path}. Please check the path.")
