"""
makes coordinates
"""

from pathlib import Path

from template6alh.matplotlib_slice import write_landmarks

write_landmarks(
    Path("template6alh/input_coords.nrrd"), Path("template6alh/man_in_coords")
)
