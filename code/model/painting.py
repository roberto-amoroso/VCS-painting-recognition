class Painting:
    """
    Class describing a painting.
    """

    def __init__(self,
                 image,
                 title,
                 author,
                 room,
                 filename,
                 frame_contour=None,
                 points=None,
                 corners=None):
        self.image = image
        self.title = title
        self.author = author
        self.room = room
        self.filename = filename
        self.frame_contour = frame_contour
        self.points = points
        self.corners = corners
