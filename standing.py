#########################################


from keras.models import load_model
import cv2
import numpy as np
import time
import os

model = load_model('model-017.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}


def MaskDetection():
    count = 0
    nomask = 0

    source=cv2.VideoCapture(0)

    while(True):

        ret,img=source.read()
        if not ret:
            break
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_clsfr.detectMultiScale(gray,1.3,5)  
        

        for (x,y,w,h) in faces:
        
            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=model.predict(reshaped)

            label=np.argmax(result,axis=1)[0]
            if label == 1 :
                #print(label)
                nomask = nomask + 1
                #print(nomask)
            
            if label == 0 :
                #print(label)
                count = count + 1
                #print(count)
                
                
        
            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            if label == 0:
                cv2.putText(img," Processing: " + str(count) +"%" , (x+10, y-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

            if label == 1:
                cv2.putText(img," Processing: " + str(nomask)+ "%" , (x+10, y-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            
            
        cv2.imshow('LIVE',img)
        if count >= 100:
            return 1
        if nomask >= 100:
            return 2
        
        if cv2.waitKey(1) == ord('q'):
            return 0
            

    cv2.destroyAllWindows()
    source.release()



def FaceDetection():

    source=cv2.VideoCapture(0)

    while(True):

        ret,img=source.read()
        if not ret:
            break
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_clsfr.detectMultiScale(gray,1.3,5)

        if(len(faces) == 1):
            print("Face Detected \n")
            print("Please Wait")
            source.release()
            time.sleep(2.0)
    
            k = MaskDetection()
            if( k == 1):
                c = 0
                print("Mask Detected")
                os.system("mpg321 mask.mp3")
                return 2

            if(k == 2):
                c = 1 
                print("No Mask Detected")
                print("Try Again")
                os.system("mpg321 nomask.mp3")
                time.sleep(2)
                FaceDetection()
                

            if(k == 0):
                print("Quit")
                os.system("mpg321 noentry.mp3")
                return 1


os.system("mpg321 good.mp3")
time.sleep(1)
FaceDetection()
