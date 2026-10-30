from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str = 'datamate') -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter('%(asctime)s %(levelname).1s %(name)s: %(message)s', '%H:%M:%S'))
        log.addHandler(h)
        log.setLevel(os.environ.get('DATAMATE_LOG', 'INFO'))
        log.propagate = False
    return log
