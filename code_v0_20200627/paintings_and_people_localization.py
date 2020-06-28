"""
Module containing functions to perform People Localization.
"""
import time


def locale_paintings_and_people(paintings_detected, generator, show_image, print_next_step, print_time):
    """Locale paintings and people in the image, assigning them to one room.

    To locate paintings and people in the image, it uses the information about the room
    where the paintings retrieved from the image are located.

    Parameters
    ----------
    paintings_detected: list
        a list containing one `Painting` object for each
        painting detected in the input image.
    generator: generator
        generator function used to take track of the current step number
        and print useful information during processing.
    show_image: function
        function used to show image of the intermediate results
    print_next_step:function
        function used to print info about current processing step
    print_time: function
        function used to print info about execution time

    Returns
    -------
    int or None
        return the number of the room or None if it's impossible to
        locale people.
    """

    # Step ZORO: People Localization
    # ----------------------------
    print_next_step(generator, "People Localization")
    start_time = time.time()
    # Choose the room of the actual video frame by majority
    major_room = None
    possible_rooms = [p.room for p in paintings_detected if p.room is not None]
    if len(possible_rooms) > 0:
        major_room = max(possible_rooms, key=possible_rooms.count)

    print_time(start_time)

    return major_room
