#%% Logging utils
import sys
import logging
import types
from pathlib import Path

def log_newline(self, how_many_lines=1):
    """
    Add a blank to the logger. 
    Accessed as a method of the logger object with `types.MethodType(log_newline, logger)`
    """
    # From https://stackoverflow.com/a/45032701
    
    # Switch formatter, output a blank line
    self.handler.setFormatter(self.blank_formatter)

    for i in range(how_many_lines):
        self.info('')

    # Switch back
    self.handler.setFormatter(self.formatter)

def create_logger(name:str,
                  level=logging.INFO, 
                  stream=sys.stdout, 
                  filename:str=None,
                  filemode:str='w',
                  propagate=False): 
    """
    Will get or create the loger with the given name, and add a method to the logger object to output a blank line.
    A handler is created, with the given level and stream (or file output).

    Parameters
    ----------
    name : str
        Name of the logger
        
    level : optional
        Level of the logger and handler. Can be a string ('info', 'debug', 'warning') or a logging level (`logging.INFO`, `logging.DEBUG`, `logging.WARNING`).
        Defaults to `logging.INFO`
        
    stream : optional
        Stream to which the logger will output. Defaults to sys.stdout
        
    filename : str, optional
        Path to the file to which the logger will output. Defaults to None
        
    filemode : str, optional
        Mode to open the file. Defaults to 'w'
        
    propagate : bool, optional
        Whether to propagate the logs to the root logger. Defaults to False
        Warning : If sets to False, the logs will not be propagated to the root logger, and will not be output by the root logger. 
        If the root logger outputs to a file, the logs will not be saved in the file, unless the logger has the same output file.
        However, if propagate is True, the logs will be output twice if the root has a different handler (once by the logger, once by the root logger)

    Returns
    -------
    logger : logging.Logger 
        Logger object 
    """
    # From https://stackoverflow.com/a/45032701 
    
    if isinstance(level, str):
        level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}[level.lower()]
    
    # Create a handler
    if filename is not None:
        path = Path(filename).parent[0] # Get the path to the directory containing the file
        path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setLevel(level)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter(fmt="%(message)s")   
    blank_formatter = logging.Formatter(fmt="")
    handler.setFormatter(formatter)

    # Create a logger, with the previously-defined handler
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Remove all handlers already associated with the logger object
    for hd in logger.handlers:
        logger.removeHandler(hd)
    logger.propagate = propagate # Prevent the logs from being propagated to the root logger that will have different handlers
    logger.addHandler(handler) # Add the handler to the logger

    # Save some data and add a method to logger object
    logger.handler = handler
    logger.formatter = formatter
    logger.blank_formatter = blank_formatter
    logger.newline = types.MethodType(log_newline, logger)
    
    return logger
