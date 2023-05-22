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

For this I have used sqlite3, Open the sqlite software and create a new database.
Create a new table with name `people` 
Next we will our first column `ID` and set it as `primary key` and datatype as `INT` because ID's will be stored in the ***integer form***
Now once we are donw with the ID column, we will create the `NAME` column with `STRING` as it datatype and choose `not null` to be it's constraints.
Similary add `AGE`,`GENDER` and `CRIMINAL ACTIVITY` column with `STRING` as it's datatype and do not choose any constraints as of now.     

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
 
 ### Step3:
 
 This step is one of the crucial steps as it contains the detection of face.
 
 As always we will begin the `detector.py` with loading some dependencies.
 
 ```
 def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
 ```
 Earler we did somewhat similar to update the items into our database but now in this we will be getting id and name from the database.
``` 
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 0, 0)
 ```
Making these variables so we don't have to write again and again them while making square around the face and to display the content related to the person we are detecting live.

`faces=faceDetect.detectMultiScale(gray,1.3,5);`
It uses the function detectMultiScale to detect the upper left corner of the rectangle (x, y) on the face as well as width and height of the rectangle. detectMultiScale function has parameters such as scaling and neighbours.

In the previous line one will get the coordinates of the rectangle. Now it is time to draw the rectangle with these coordinates on the colored image (frame). For this we will make the use of cv2.rectangle where we can specify color of rectangle as well as width of the bounding box. This is executed in a loop.

```
for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        id,conf=rec.predict(gray [y:y+h , x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(img,str(profile[1]),(x,y+h+30),fontFace,fontScale,fontColor)
            cv2.putText(img,str(profile[2]),(x,y+h+60),fontFace,fontScale,fontColor)
            cv2.putText(img,str(profile[3]),(x,y+h+90),fontFace,fontScale,fontColor)
            cv2.putText(img,str(profile[4]),(x,y+h+120),fontFace,fontScale,fontColor)
        # cv2.putText(img,str(id),(x,y+h),font,2,(255))
        # cv2.cv.putText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(255))
    cv2.imshow("Face",img);
```
```
cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
id,conf=rec.predict(gray [y:y+h , x:x+w])
 ```
Coordinates of the rectangle for the face is calculated. In this line such area is sliced from the grayscale image.
Face area for the colored image is sliced.

Then if the profile is not equal to none, we shall display all the information about that person.

Now to close the window press 'q' to quit it.

The below screenshot is `1.png` from the `output_screenshot` folder. This screenshot is from the time when only 'name' and 'id' was added in the database.
If you've followed the above steps correctly then it should display all four of the details we added in the database.

![1](https://github.com/nishit-hirani/real-time-FaceRecognition-using-OpenCV-with-database-and-flask-implementation/assets/89455398/a299ae29-80e9-4d8d-8164-6ea550beb938)


##### Till this step you will be able to detect faces and display information, but it has not been uploaded to the web yet for many till this step it would be enough but for those who want to take it to web proceed with the steps below 



