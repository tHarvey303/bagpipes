import importlib
import sys
import os

def set_config(config_name, return_config=False, reload=True):
    """
    Sets a new configuration module.

    Properly updates all config references across the bagpipes package to ensure
    all submodules use the new configuration. This handles all the various ways
    the config can be referenced:
    - from bagpipes import config  
    - bagpipes.config
    - bagpipes.configs.config
    """
    os.environ['PIPES_CONFIG_NAME'] = config_name
    
    try:
        # The module name is relative to the current package ('bagpipes')
        if config_name != '':
            if config_name[0] != '_':
                config_name = '_' + config_name

        module_name = '.configs.config' + config_name
        config_module = importlib.import_module(module_name, package='bagpipes')

    except ImportError:
        # Handle cases where the specified config doesn't exist
        print(f"Warning: Configuration '{module_name}' not found. Falling back to default.")
        config_module = importlib.import_module('.configs.config_BC03', package='bagpipes')

    # Update all possible config references in sys.modules to ensure consistency
    # This handles: bagpipes.config, bagpipes.configs.config
    sys.modules['bagpipes.config'] = config_module
    sys.modules['bagpipes.configs.config'] = config_module
    
    # Update the main bagpipes module's config attribute if it exists
    if 'bagpipes' in sys.modules:
        sys.modules['bagpipes'].config = config_module
    
    # For modules that use "from bagpipes import config", we need to update
    # their local reference. We do this by finding modules that have imported
    # bagpipes and updating their config attribute.
    for module_name, module in list(sys.modules.items()):
        if module is not None and hasattr(module, '__dict__'):
            # Check if this module has imported config from bagpipes
            if (hasattr(module, 'config') and 
                module_name.startswith('bagpipes.') and
                'config' not in module_name and  # Don't update config modules themselves
                hasattr(module.config, '__name__') and
                'bagpipes.configs.config' in str(module.config.__name__)):
                
                # Update the module's config reference
                module.config = config_module

    if return_config:
        return config_module
