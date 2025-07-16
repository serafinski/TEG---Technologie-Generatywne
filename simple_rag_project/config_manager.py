import os 
import yaml
from typing import Dict, Any
from copy import deepcopy

class ConfigManager:
    def __init__(self, config_path: str = None):
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(current_dir, "config.yaml")
        else:
            self.config_path = config_path
            
        print(f"Looking for config file at: {self.config_path}")
        self.default_config = self._load_config()
        if self.default_config is None:
            raise ValueError(f"Failed to load configuration from {self.config_path}")
        
        # Create a runtime config that can be modified during the session
        self.runtime_config = deepcopy(self.default_config)

    def _load_config(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    loaded_config = yaml.safe_load(file)
                    if loaded_config is None:
                        print(f"Error: Config file {self.config_path} exists but is empty or invalid")
                        return None
                    print(f"Successfully loaded config from {self.config_path}")
                    return loaded_config
            else:
                print(f"Error: Config file not found at {self.config_path}")
                return None
        except Exception as e:
            print(f"Error loading config file: {e}")
            return None

    def get_config(self) -> Dict[str, Any]:
        return self.runtime_config
    
    def get_default_config(self) -> Dict[str, Any]:
        return self.default_config
    
    def reset_to_defaults(self):
        """Reset runtime config to the original defaults"""
        try:
            # Create a fresh deep copy to ensure complete reset
            self.runtime_config = deepcopy(self.default_config)
            return True
        except Exception as e:
            print(f"Error resetting to defaults: {e}")
            return False
    
    def update_runtime_config(self, key: str, value: Any):
        """Update a value in the runtime config without changing the default config"""
        keys = key.split('.')
        config = self.runtime_config
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Update the value
        config[keys[-1]] = value
    
    def get_value(self, key: str, use_default: bool = False) -> Any:
        """Get a value from either runtime or default config"""
        config = self.default_config if use_default else self.runtime_config
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except Exception as e:
            print(f"Error retrieving value for key '{key}': {e}")
            return None