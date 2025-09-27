import importlib
import sys
import os
import glob

def set_config(config_name, return_config=False, reload=True):
    """
    Sets a new configuration module.

    Properly updates all config references across the bagpipes package to ensure
    all submodules use the new configuration. This handles all the various ways
    the config can be referenced:
    - from bagpipes import config  
    - bagpipes.config
    - bagpipes.configs.config

    Parameters
    ----------
    config_name : str
        Name of the configuration to load. Can be with or without leading underscore.
        Examples: 'BC03', '_BC03', 'bpass', 'test'
    return_config : bool, optional
        If True, returns the loaded config module. Default False.
    reload : bool, optional
        Legacy parameter, kept for compatibility. Default True.

    Returns
    -------
    module or None
        The loaded configuration module if return_config=True, otherwise None.

    Examples
    --------
    >>> from bagpipes.config_utils import set_config
    >>> # Switch to BPASS configuration
    >>> set_config('bpass')
    >>> # Switch to custom configuration and get reference
    >>> my_config = set_config('my_custom_config', return_config=True)
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


def list_available_configs():
    """
    List all available configuration modules.
    
    Returns
    -------
    list of str
        List of available configuration names (without 'config_' prefix and '.py' suffix).
        
    Examples
    --------
    >>> from bagpipes.config_utils import list_available_configs
    >>> configs = list_available_configs()
    >>> print("Available configs:", configs)
    """
    try:
        # Get the configs directory path
        import bagpipes.configs
        configs_dir = os.path.dirname(bagpipes.configs.__file__)
        
        # Find all config_*.py files
        config_files = glob.glob(os.path.join(configs_dir, 'config_*.py'))
        
        # Extract config names (remove config_ prefix and .py suffix)
        config_names = []
        for config_file in config_files:
            filename = os.path.basename(config_file)
            if filename.startswith('config_') and filename.endswith('.py'):
                config_name = filename[7:-3]  # Remove 'config_' and '.py'
                config_names.append(config_name)
        
        return sorted(config_names)
        
    except Exception as e:
        print(f"Warning: Could not list available configs: {e}")
        return []


def get_current_config():
    """
    Get the currently active configuration module.
    
    Returns
    -------
    module
        The currently active configuration module.
        
    Examples
    --------
    >>> from bagpipes.config_utils import get_current_config
    >>> current = get_current_config()
    >>> print(f"Current config: {current.__name__}")
    """
    if 'bagpipes' in sys.modules and hasattr(sys.modules['bagpipes'], 'config'):
        return sys.modules['bagpipes'].config
    elif 'bagpipes.config' in sys.modules:
        return sys.modules['bagpipes.config']
    else:
        return None


def validate_config(config_module):
    """
    Validate that a configuration module has required attributes.
    
    Parameters
    ----------
    config_module : module
        The configuration module to validate.
        
    Returns
    -------
    list of str
        List of missing required attributes, empty if all are present.
    """
    required_attrs = ['max_redshift', 'R_spec', 'R_phot', 'R_other']
    missing_attrs = []
    
    for attr in required_attrs:
        if not hasattr(config_module, attr):
            missing_attrs.append(attr)
    
    return missing_attrs


def reload_config_from_environment():
    """
    Reload configuration based on environment variables.
    
    This function checks the standard bagpipes environment variables 
    and reloads the appropriate configuration:
    - PIPES_CONFIG_NAME: Explicit config name
    - use_bpass: If set to '1', uses bpass config
    
    Returns
    -------
    module
        The loaded configuration module.
        
    Examples
    --------
    >>> import os
    >>> os.environ['PIPES_CONFIG_NAME'] = 'test'
    >>> from bagpipes.config_utils import reload_config_from_environment
    >>> config = reload_config_from_environment()
    """
    config_name = 'BC03'  # default
    
    try:
        use_bpass = bool(int(os.environ['use_bpass']))
        if use_bpass:
            config_name = '_bpass'
    except (KeyError, ValueError):
        try:
            config_name = os.environ['PIPES_CONFIG_NAME']
            if not config_name.startswith('_'):
                config_name = '_' + config_name
        except KeyError:
            pass
    
    return set_config(config_name, return_config=True, reload=False)
