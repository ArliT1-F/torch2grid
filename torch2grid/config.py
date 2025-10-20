import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    # Visualization settings
    'output_dir': 'grids',
    'figure_size': (6, 6),
    'colormap': 'viridis',
    'interpolation': 'nearest',
    'dpi': 100,
    
    # Dead neuron detection
    'dead_neuron_threshold': 1e-6,
    'dead_neuron_warning_threshold': 0.1,  # 10% dead neurons triggers warning
    
    # Gradient analysis
    'gradient_vanishing_threshold': 1e-7,
    'gradient_exploding_threshold': 1e3,
    
    # Export settings
    'export_formats': ['png'],
    'export_quality': 95,
    
    # Plugin settings
    'plugin_directory': 'plugins',
    'auto_load_plugins': True,
}

def get_config() -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables
    for key, value in config.items():
        env_key = f"TORCH2GRID_{key.upper()}"
        if env_key in os.environ:
            env_value = os.environ[env_key]
            # Try to convert to appropriate type
            if isinstance(value, bool):
                config[key] = env_value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(value, int):
                try:
                    config[key] = int(env_value)
                except ValueError:
                    pass
            elif isinstance(value, float):
                try:
                    config[key] = float(env_value)
                except ValueError:
                    pass
            elif isinstance(value, str):
                config[key] = env_value
            elif isinstance(value, list):
                config[key] = [item.strip() for item in env_value.split(',')]
    return config

def set_config(key: str, value: Any) -> None:
    if key not in DEFAULT_CONFIG:
        raise KeyError(f"Unknown configuration key: {key}")
    
    # Update the global config (this is a simple implementation)
    # In a more sophisticated setup, you might want to use a proper config manager
    global _current_config
    if '_current_config' not in globals():
        _current_config = get_config()
    _current_config[key] = value

def get_config_value(key: str, default: Any = None) -> Any:
    config = get_config()
    return config.get(key, default)