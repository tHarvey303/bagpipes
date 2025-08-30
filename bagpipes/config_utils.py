import importlib
import sys
import os

def set_config(config_name, return_config=False, reload=True):
    """
    Sets a new configuration module.

    Reloads the specified configuration module and updates the global config variable.
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

    # Update the global config variable
    config = config_module

    # Make sure other modules are correct- e.g. bagpipes.configs.config in sys.modules
    sys.modules['bagpipes.configs.config'] = config
    sys.modules['bagpipes.config'] = config

    for k,v in list(sys.modules.items()):
        if k.startswith('bagpipes') and 'config' not in k:
            #print(f"Reloading module: {k}")
            importlib.reload(v)
    
    #if reload:
    #    # Reload the module to ensure any changes are applied
    importlib.reload(sys.modules['bagpipes'])


    if return_config:
        return config
