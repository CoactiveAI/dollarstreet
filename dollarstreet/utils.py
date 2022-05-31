from datetime import datetime
import functools
import logging
import os
import traceback
from typing import Callable, Optional
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.hist.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pil_loader(path: str) -> Image.Image:
    """Loads images from path as PIL object in RGB format.
    """
    img = Image.open(path)
    return img.convert("RGB")


def log(
    _func: Callable = None, 
    *,
    logger: Optional[logging.Logger] = None,
    log_input: Optional[bool] = True,
    log_output: Optional[bool] = True,
) -> Callable:
    """Decorator function. Initialized Logger can be provided. Decorated
        functions should use save_log:bool flag to indicate whether results
        should be saved to disk.
    """
    # Get logger
    if not logger:
        logging.basicConfig(
            format='%(levelname)s:%(message)s', level=logging.DEBUG)
        logger = logging.getLogger(__name__)

    def decorator_log(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):

            # Set start time
            start_dt = datetime.now()

            # Add file handler if save_log flag is set
            save_log = kwargs.get('save_log', False)
            if save_log:

                log_save_dir = 'logs'
                if not os.path.exists(log_save_dir):
                    os.makedirs(log_save_dir)

                formatted_start_dt = f'{start_dt:%Y-%m-%d_%Hh_%Mm_%Ss_%fms}'
                filename = f'{func.__name__}_{formatted_start_dt}.log'
                save_path = os.path.join(log_save_dir, filename)
                handler = logging.FileHandler(save_path)
                handler.setLevel(logging.DEBUG)
                logger.addHandler(handler)

            # Log function, args and start time
            logger.info(f'Function: {func.__name__}')
            if log_input:
                logger.info(f'args: {args}')
                logger.info(f'kwargs {kwargs}\n')
            logger.info(f'---------- Run at {start_dt}\n')

            # Try to run. Log if error is thrown
            try:
                # Get output
                output = func(*args, **kwargs)

                # Log outut
                if log_output:
                    end_dt = datetime.now()
                    logger.info('---------- Output')
                    logger.info(f'{output}')

                # Log end time
                end_dt = datetime.now()
                logger.info(f'---------- Finished at {str(end_dt)}')
                logger.info(f'Run time: {str(end_dt - start_dt)}')

                return output

            except Exception as e:
                # Log error
                logger.error(traceback.format_exc())

                # Log end time
                end_dt = datetime.now()
                logger.info(f'---------- Finished at {str(end_dt)}')
                logger.info(f'Run time: {str(end_dt - start_dt)}')

                raise e

        return wrapped

    if _func is None:
        return decorator_log
    else:
        return decorator_log(_func)
