"""
Class containing information about a Painting
"""


class Painting:
    """
    Class describing a painting.
    """

    def __init__(self,
                 image=None,
                 title=None,
                 author=None,
                 room=None,
                 filename=None,
                 bounding_box=None,
                 frame_contour=None,
                 points=None,
                 corners=None):
        # It will contain the DB image or the rectified sub-image (up-scaled if necessary)
        self.image = image
        self.title = title
        self.author = author
        self.room = room
        self.filename = filename
        self.frame_contour = frame_contour
        self.points = points
        self.corners = corners
        self.bounding_box = bounding_box
