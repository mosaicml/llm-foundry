import logging 

def configure_logging(logging_level: str, log: logging.Logger):
    """Configure logging.

    Args:
        logging_level (str): Logging level.
    """
    logging.basicConfig(
        format=
        f'%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
    )
    logging_level = logging_level.upper()
    logging.getLogger('llmfoundry').setLevel(logging_level)
    logging.getLogger(__name__).setLevel(logging_level)
    log.info(f'Logging level set to {logging_level}')
    