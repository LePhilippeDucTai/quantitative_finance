import sys

from loguru import logger

_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"  # noqa: E501
config = {"handlers": [{"sink": sys.stdout, "format": _format}]}

logger.remove()
logger.configure(**config)
