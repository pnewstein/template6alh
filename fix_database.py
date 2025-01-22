"""
This script can be used to get a good interactive environment to fix the database
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from template6alh.sql_classes import RawFile, Channel, Image
from template6alh.utils import get_engine

session = Session(get_engine())

channels = session.execute(select(Channel)).scalars().all()
