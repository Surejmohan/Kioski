# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

from flask import Response
from flask import Flask
from flask import render_template,redirect,url_for
import threading
import datetime
import time
from keras.models import load_model
import cv2
import numpy as np
import time
import datetime
import os

model = load_model('model-017.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
nomaskcount=0
call=0

lock = threading.Lock()
app = Flask(__name__)



def MaskDetection():
    count = 0
    nomask = 0

    source=cv2.VideoCapture(0)

    while(True):

        ret,img=source.read()
        if not ret:
            break
        img = cv2.resize(img,(600,600), interpolation =cv2.INTER_AREA)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_clsfr.detectMultiScale(gray,1.3,5)  
        
        timestamp = datetime.datetime.now()
        cv2.putText(img, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        if count >= 98:
            cv2.putText(img,"Mask found" , (int(img.shape[1]/2)-5,int(img.shape[0]/4)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        if nomask >= 98:
            cv2.putText(img,"No Mask found" , (int(img.shape[1]/2)-5,int(img.shape[0]/4)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        for (x,y,w,h) in faces:
        
            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=model.predict(reshaped)

            label=np.argmax(result,axis=1)[0]
            if label == 1 :
                #print(label)
                nomask = nomask + 2
                #print(nomask)
            
            if label == 0 :
                #print(label)
                count = count + 2
                #print(count)
                
                
        
            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            if label == 0:
                cv2.putText(img," Processing: mask found " + str(count) +"%" , (x, y-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

            if label == 1:
                cv2.putText(img," Processing:mask not found " + str(nomask)+ "%" , (x, y-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            
        if count >= 101:
            return 1
        if nomask >= 101:
            return 2

        cv2.imshow('LIVE',img)
        
        
        if cv2.waitKey(1) == ord('q'):
            return 0
            

    cv2.destroyAllWindows()
    source.release()



def FaceDetection():
    global nomaskcount,call
    if (call==0):
        os.system('mpg321 welcome1.mp3')
        call=1


    source=cv2.VideoCapture(0)
    

    while(True):

        ret,img=source.read()
        if not ret:
            break
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_clsfr.detectMultiScale(gray,1.3,5)

        if(len(faces) == 1):
            os.system("mpg321 welcomenote.mp3")
            print("Face Detected \n")
            print("Please Wait")
            source.release()
            time.sleep(2.0)

            k = MaskDetection()
            if( k == 1):
                
                print("Mask Detected")
                os.system("mpg321 mask.mp3")
                cv2.destroyAllWindows()
                return 2

            if(k == 2 and nomaskcount<3):
                 
                print("No Mask Detected")
                print("Try Again")
                os.system("mpg321 nomask.mp3")
                cv2.destroyAllWindows()
                nomaskcount=nomaskcount+1
                if(nomaskcount==3):
                    print("Mask not found.Your 3 attempt failed. You cant enter")
                    os.system("mpg321 noentry.mp3")
                    cv2.destroyAllWindows()
                    return 1
                time.sleep(2)
                FaceDetection()
            if(k == 0):
                print("Quit")
                os.system("mpg321 noentry.mp3")
                cv2.destroyAllWindows()
                return 1



def Main():
    FaceDetection()
    time.sleep(7.0)
    return Main()
        


@app.route("/")
def index():
    t = threading.Thread(target=Main)
    t.daemon = True
    t.start()
    return render_template("slide.html")

    



if __name__ == '__main__':
    import random, threading, webbrowser
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    threading.Timer(1.25, lambda: webbrowser.open(url) ).start()

    app.run(port=port, debug=False)
    
   
   # app.run(host='0.0.0.0', port='8000', debug=True,
      #  threaded=True, use_reloader=False)

