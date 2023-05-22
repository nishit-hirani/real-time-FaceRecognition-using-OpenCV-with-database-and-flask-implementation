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

```
def insertOrUpdate(Id,Name):
    conn = sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1;
        
    if(isRecordExist==1):
        cmd="UPDATE People SET Name="+str(Name)+"WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO People(ID,Name) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
```
Next step I did was to create a fucntion `insertorUpdate` which establishes a connection between the database and the .py file and upadating the table in the databsae when user gives a valid input.

```
id=input("enter id")
name=input("enter name")
```
A simple py code to ask the user to input the data.
Note: When you enter a name to be stored in database in the database, enter the name in ***double quotes***

```
sampleNum=0
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.waitKey(1000);
    cv2.imshow("Face",img);
    cv2.waitKey(1)
    if(sampleNum>20):
        break;
cam.release()
cv2.destroyAllWindows()
```
The next portion of the code converts the taken images into gray scale and detect faces using the .xml file and in the for loop we are storing the images the ***dataSet*** folder with the file name as User.1.(followed by the id number given earlier when creating a database.
![image](https://github.com/nishit-hirani/real-time-FaceRecognition-using-OpenCV-with-database-and-flask-implementation/assets/89455398/ca722229-7dd9-4bba-a216-7786769af258)

The picture will be taken within a time interval of 1s and here we are taking max 21 pictures. The more the picture the better will be accuracy.
Next we are destroying all the windows.

### Step2:

Now we shall make a file `trainer.py` to train the data we have collected earlier.

First we will load all the dependencies, then we will make this function to get the images along with the face ID and go through them to train the data.

```
def getImageWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L');
        faceNp = np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print (ID)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return IDs,faces
 ```
 
While the data is being trained you will see a small where all the images will be iterated and on the console you will see the ID nummber simultaneously.

`recognizer.save('recognizer/trainingData.yml')`
After the data has been trained the data will be saved as `trainingData.yml` in the `recognizer` folder
 
