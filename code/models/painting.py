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
                 corners=None,
                 keypoints=None,
                 descriptors=None):
        """Constructor of the Painting class.

        Parameters
        ----------
        image: ndarray
            It will contain the DB image or the rectified sub-image (up-scaled if necessary)
        title: str
            Title of the painting
        author: str
            Author of the painting
        room: int
            number of the romm where the painting is located
        filename: str
            name of the file in the DB asspciated to the painting
        bounding_box: ndarray
            array [x, y, w, h] representing the bounding box of the painting
        frame_contour: list
            list of points (x, y) representing the contours of the painting including the frame
        points: list
            list of points (x, y) representing the contours of the painting without the frame
        corners: ndarray
            array of point [x,y] representing the corners of the painting
        keypoints: list
            ORB keypoints for the Painting Retrieval Task
        descriptors: list
            ORB descriptors for the Painting Retrieval Task
        """

        self.image = image
        self.title = title
        self.author = author
        self.room = room
        self.filename = filename
        self.frame_contour = frame_contour
        self.points = points
        self.corners = corners
        self.bounding_box = bounding_box
        self.keypoints = keypoints
        self.descriptors = descriptors
