# real-time-FaceRecognition-using-OpenCV-with-database-and-flask-implementation
This projects is about face recognition which is connected with a backend and is implemented on web using flask

### Dependencies to run the project:
- pip install Flask
- pip install db-sqlite3
- pip install opencv-python
- pip install numpy

OpenCV will be used to draw the rectangle on the face. OpenCv as well as haar cascade frontal face xml files will be used to locate the coordinates of the face.
This algorithm is composed of two parts – training as well as detection. It not only detects faces in the images but is applicable to videos also. There is a limitation of this algorithm as it works with only frontal faces which is overcomed by deep learning models such as ssd, yolo, faster rcnn.

### The Haar-Cascade Face Detection Algorithm : 
The Haar-Cascade Face Detection Algorithm is a sliding-window type of algorithm that detects objects based upon its features. 
### Haar Face Features:
The Haar-Cascade model, employs different types of feature recognition that include the likes of

Size and location of certain facial features. To be specific, nose bridge, mouth line and eyes.
- Eye region being darker than upper-cheek region.
- Nose bridge regio being brighter than eye region.

Intel's 'haarcascade_frontalface_default.xml'
This 'XML' file contains a pre-trained model that was created through extensive training and uploaded by Rainer Lienhart on behalf of Intel in 2000.
