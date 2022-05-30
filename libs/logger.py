import logging

_FORMAT = "%(asctime)s %(funcName)s %(levelname)s - %(message)s"
logging.basicConfig(format=_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)
