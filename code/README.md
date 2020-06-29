
# VCS 2020: Locate and Recognize Paintings and People

This document describes the structure and functionality of the software I created as the final project of the course of Vision and Cognitive Systems AY 2019/2020.

I propose a method to locate and recognize paintings and people in a museum or art gallery. For this purpose, I created a ***Python*** program that is able to locate and recognize paintings and people present in a video or single image. For the part relating to the paintings, I used the ***OpenCV*** library, while to carry out the people detection operation I used [***YOLO***](https://pjreddie.com/darknet/yolo/), a real-time object detection system.

Before proceeding to the detailed description of the functionalities of the program and the possible arguments accepted, the prototype invocation of the main script follows:

    $ python main_detector.py [-h] [-o OUTPUT] [-t {0,1,2,3,4,5}]
                            [-fo FRAME_OCCURRENCE] [-vp {0,1}] [-vi {0,1,2}]
                            [-mdbi] [-hm]
                            input_filename db_path data_filename

The program, calleble through the script `main_detector.py`,  allows you to perform the following tasks:

- **Painting Detection**:  detects all paintings.
- **Painting Segmentation**:  creates a segmented version of the input, where the paintings, and also any statues, identified are white and the background is black.
- **Painting Rectification**:  rectifies each painting detected, through an affine transformation
- **Painting Retrieval**: matches each detected and rectified painting to the paintings DB found in `db_path`
- **People Detection**: detects people in the input
- **People and Painting Localization**: locates paintings and people using information found in `data_filename`


## Code Documentation
The documentation relating to the code implemented, with a wide description of all functions, classes, modules contained in them, is available by accessing the `Documentation.html` file.  
It is an HTML interface that makes it easy and immediate to consult the documentation and navigate between the various modules within the project.

For the realization of the documentation I used [pdoc](https://pdoc3.github.io/pdoc/), a software package for the generation of API documentation for Python.

## Preamble

It is advisable to set up a virtual environment in which to install all the necessary packages. These are indicated in the `requirements.txt` file inside the project folder.

**NOTE**:  the following packages are present in the `requirements.txt` file:

>torch\==1.5.1+cu101  
>torchvision\==0.6.1+cu101

They install ***PyTorch***, necessary to use ***YOLOv3***. The version I installed works well for a system with ***CUDA 10.1***. 
To avoid errors and malfunctions, please install the version suitable for your environment by following the instructions on the official page of [PyTorch](https://pytorch.org/).

## Virtual Environment and Packages

To get started, you’ll want to install the `virtualenv` tool with `pip`:

    $ pip install virtualenv

Let's create a virtual environment:

    $ python -m virtualenv <venv_name>

Activate the previously created virtual environment:

    $ source <venv_name>/bin/activate
    
Now we use `pip` to install the packages of the `requirements.txt` file:

    $ pip install -r requirements.txt
    
Now we have prepared the virtual environment to run the program.

## Usage

All the information in this document can be accessed directly by viewing the `main_detector.py` script help message via the `-h` or `--help` argument.

For convenience, we report again the program invocation prototype:

    $ python main_detector.py [-h] [-o OUTPUT] [-t {0,1,2,3,4,5}]
                            [-fo FRAME_OCCURRENCE] [-vp {0,1}] [-vi {0,1,2}]
                            [-mdbi] [-hm]
                            input_filename db_path data_filename

### Positional Arguments:

 - `input_filename`        filename of the input image or video
 - `db_path`     path of the directory where the images that make up  the DB are located
 - `data_filename`       file containing all the information about the  paintings: *(Title, Author, Room, Image)*

### Optional Arguments:

 - `-h, --help`<br> 
 show an help message and exit

 - `-o OUTPUT, --output OUTPUT`<br> 
path used as base to determine where the outputs are stored. For details, read the epilogue at the bottom, section `# OUTPUT`

 - `-t {0,1,2,3,4,5}, --task {0,1,2,3,4,5}`<br> 
determines which task will be performed on the input.<br> 
***NOTE***: for details on how the tasks are performed and for some examples, read  the epilogue at the bottom of the page, section `# TASKS`<br> 
                          0 = Painting Detection<br> 
                          1 = Painting Segmentation<br> 
                          2 = Painting Rectification<br> 
                          3 = Painting Retrieval<br> 
                          4 = People Detection<br> 
                          5 = People and Paintings Localization **(default)**

 - `-fo FRAME_OCCURRENCE, --frame_occurrence FRAME_OCCURRENCE`<br> 
integer >=1 (default =1). In case the input is a video, it establishes with which occurrence to consider the frames of the video itself.<br> 
***Example***: `frame_occurrence = 30` (**value recommended during debugging**) means that it considers one frame every 30.<br> 
***NOTE***: for more details read the epilogue at the bottom of the page, section `# FRAME_OCCURRENCE`

 - `-vp {0,1}, --verbosity_print {0,1}`<br> 
set the verbosity of the information displayed (description of the operation  executed and its execution time)<br> 
                          0 = ONLY main processing steps info **(default)**<br> 
                          1 = ALL processing steps info<br> 

 - `-vi {0,1,2}, --verbosity_image {0,1,2}`<br> 
set the verbosity of the images displayed.<br> 
***NOTE***: if the input is a video, is automatically set to '0' (in order to avoid  an excessive number of images displayed on the screen).<br> 
                          0 = no image shown **(default)**<br> 
                          1 = shows main steps final images, at the end of the script execution ***(NOT BLOCKING)***<br> 
                          2 = shows each final and intermediate image when it is created and a button
                              must be pressed to continue the execution (mainly used for debugging) ***(BLOCKING)***

 - `-mdbi, --match_db_image`<br> 
 if present, to perform *Painting Retrieval*, the program rectifies each painting  to match the aspect ration of every painting in `db_path`. Otherwise, it rectifies each painting one time using a calculated aspect ratio.

 - `-hm, --histo_mode`<br> 
if present indicates that, during *Painting Retrieval*, the program will executes  *Histogram Matching* in the case *ORB* does not produce any match.<br> 
***WARNINGS***: setting `--histo_mode` increases the percentage of matches with the DB, but decreases the precision, i.e. increases the number of false positives (incorrect matches).



## Epilogue and Examples

### \# TASKS:

Given the mutual dependency of the tasks, to execute the $i-th$ task, with $i>1$, it is necessary that the j-th tasks are executed first, for each $j$ such that $0<=j<i$.
For example, if you want to perform *Painting Rectification* ($i = 2$) it is necessary that you first execute *Painting Segmentation* ($j = 1$) and *Painting Detection* ($j = 0$).
The *People Detection* task is an **exception**. It runs independently of the other tasks.

### \# OUTPUT:
   
 the program output paths are structured as follows (let's consider `--output = "output"`):

                output/
                 |-- painting_detection/
                 |-- painting_segmentation/
                 |-- painting_rectification/
                 |  |-- <input_filename>/
                 |-- painting_retrieval/
                 |-- people_detection/
                 |-- paintings_and_people_localization/

Each sub-directory will contain the output of the related task (indicated by the name of the sub-directory itself). The output filename will be the same of the input (`input_filename`).

The type of the output follows that of the input: `image -> image` and `video -> video`. The exception is the *Painting Rectification* task, which produces only images as output, specifically one image for each individual painting detected. 

It is clear that the number of images produced can be very high, especially in the case of videos. To improve the organization and access to data, the rectified images produced are stored in a directory that has the same as `input_filename`. Inside this directory, the images are named as follows (the extension is neglected):

      input = image -> '<input_filename>_NN' where NN is a progressive number assigned to each
                        painting found in the image.
      input = video -> '<input_filename>_FFFFF_NN' where NN has the same meaning as before but
                        applied to each video frame, while FFFFF is a progressive number assigned
                        to each frame of the video that is processed.

### \# FRAME_OCCURRENCE:

in case `--frame_occurrence` is $> 1$, the frame rate of the output video will be set so that it has the same duration as the input video.

### \# EXAMPLE:
    
A full example could be:
    

    $ python main_detector.py dataset/videos/014/VID_20180529_112627.mp4 painting_db/ data/data.csv -o output -t 5 -fo 30 -vp 1 -vi 2 --match_db_image --histo_mode
    
 In this case, the input is a video and we want to perform the *Painting and People Localization* task. This implies that all tasks (0 to 5) will be performed. The video will be processed  considering one frame every 30 occurrences. 
 
 All intermediate results will be printed, but no  image will be displayed during processing because we are working with a video and `-vi`  is automatically set equal to 0 (read `-vi` for details). 
 
 The rectification of each detected  painting will be carried out to match the aspect ratio of each image of the DB. 
 In the event that ORB does not produce any match, a match based on histogram will be executed. 
 
 The output is a video stored in `'./output/paintings_and_people_localization/VID_20180529_112627.mp4'` whose frames show the results of the tasks performed on the frames of the input video.

