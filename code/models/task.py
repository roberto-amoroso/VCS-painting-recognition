"""
Enum containing information about the Task to be executed
"""

from enum import Enum


class Task(Enum):
    painting_detection = 0
    # OPTIONAL
    painting_segmentation = 1
    painting_rectification = 2
    painting_retrieval = 3
    people_detection = 4
    paintings_and_people_localization = 5
    # Execute everything (not needed, at least for now...)
    # all_in = 6
