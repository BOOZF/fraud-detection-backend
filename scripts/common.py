import os
from pathlib import Path

from dotenv import load_dotenv
from teradataml import create_context

ROOT = Path(__file__).resolve().parent.parent


def connect():
    load_dotenv(ROOT / ".env")
    return create_context(
        host=os.environ["TD_HOST"],
        username=os.environ["TD_USER"],
        password=os.environ["TD_PASSWORD"],
    )
