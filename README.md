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

### STEP 1:
Is to create a `dataset_creator.py` which will save images captured from camera into a folder in the local directory and the id of the person along with there name in the database `FaceBase.db`.
After importing the libraries, we shall begin writing the code.
```
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```
Loading the cascade for face using cv2 function CascadeClassifier
```
cam = cv2.VideoCapture(0)
```
Create a cam variable to get camera access, for my web cam I passed the parameter as 0 (zero) but you can try values 1,2 and 3 or any other number according to your webcam
