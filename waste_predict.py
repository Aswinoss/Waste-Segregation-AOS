import cv2
import os
import tensorflow as tf

model=tf.keras.models.load_model("waste_segregation.model")
DIVISION=["Non-Biodegradable","Biodegradable"]
SIZE=70
def process(path):
    img_array=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img_array_new=cv2.resize(img_array,(SIZE,SIZE))
    n = (img_array_new.reshape(-1,SIZE,SIZE,1))/255
    return n

#variables for testing average prediction for a particular classification 
count=0
count1=0
count2=0

root="A:\BITS\AOS project\image classifier\Test" #dir of testing docs
for img in (os.listdir(root)):
    pred=model.predict([process(os.path.join(root,img))])
    count=count+1       
    if(pred <= 0.7): #adjusting sigmoid since images in recyclable dataset is approx(300) less
        print("Non-Biodegradable")
        count1=count1+1
    else:
        print("Biodegradable")
        count2=count2+1


#code for measuring accuracy for a particular class       
 
'''print(count)
print(count1)
#print(count2)
print(count1/count)
#print(count2/count)
'''

#getting accuracy of about 79.37% for biodegradable
#getting accuracy of about 77.68% for Non-biodegradable
#pred=model.predict([process(path)])
#print(pred)
#print(DIVISION[(pred[0][0])])