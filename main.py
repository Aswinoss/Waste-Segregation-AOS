import sys
import cv2
import os
import tensorflow as tf

from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QIcon, QFont, QPixmap, QPalette
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton


class Window(QMainWindow): 
    def __init__(self, width, height):
        super().__init__()
        self.title = "Waste Segregation"
        self.iconName = "waste.jpg"
        self.top = 10
        self.left = 10
        self.height = height
        self.width = width
        self.flag=None
        self.obt_flag=None
        self.InitWindow()

# draw UI components on the main window
    def InitWindow(self):
        self.setWindowIcon(QIcon(self.iconName))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-image: url(bg.jpg); background-position: center;")

        self.UiComponents()

        self.show()

    def UiComponents(self):
        initTop = 45
        initLeft = 790

        btnWidth = 150
        btnHeight = 50
        marginTop = 20

        self.topLabel = QLabel("Waste Segregator", self)
        self.topLabel.setFont(QFont("Helvetica", 24))
        width = 800
        self.topLabel.setGeometry(self.width // 2 - width / 2, marginTop, width, 50)
        self.topLabel.setAlignment(Qt.AlignCenter)
        self.topLabel.setAttribute(Qt.WA_TranslucentBackground)
        self.topLabel.setStyleSheet("color: white;")

        width = 300
        self.label = QLabel(self)
        self.label.setGeometry(self.width // 2 - width / 2, 100, width, width)
        self.label.setStyleSheet("background: white; "
                                 "background-position: center;"
                                 "background-image: url(robot.png) ;"
                                 "border-radius:25px;")
        width = 600
        self.label2 = QLabel(self)
        self.label2.setGeometry(self.width // 2 - width / 2, 440,width, 50)

        self.label2.setFont(QFont("Helvetica", 18))
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setText("What kind of waste do you have?")
        self.label2.setStyleSheet("border-radius:25px;"
                                  "background: white; ")

        marginTop=90

        self.organicBtn = QPushButton("", self)
        self.organicBtn.setGeometry(QRect(self.width // 2 - btnWidth - 50, 540 , btnWidth, btnHeight))
        self.organicBtn.setIconSize(QSize(50, 50))
        self.organicBtn.setText("Organic")
        self.organicBtn.clicked.connect(self.organicBtnClick)
        self.organicBtn.setStyleSheet(
            "color: #ffffff;background: #28a745;border: 2px solid #28a745;border-radius:25px;")

        self.inorganicBtn = QPushButton("", self)
        self.inorganicBtn.setGeometry(
            QRect(self.width // 2 + btnWidth - 100, 540, btnWidth, btnHeight))
        self.inorganicBtn.setIconSize(QSize(50, 50))
        self.inorganicBtn.setText("Inorganic")
        self.inorganicBtn.clicked.connect(self.inorganicBtnClick)
        self.inorganicBtn.setStyleSheet(
            "color: #ffffff;background: #1d2124;border: 2px solid #1d2124;border-radius:25px;")

        self.okBtn = QPushButton("", self)
        self.okBtn.setGeometry(QRect(self.width // 2 - 80 /2, 540, 50, 50))
        self.okBtn.setIconSize(QSize(50,50))
        self.okBtn.setIcon(QIcon("ok.png"))
        self.okBtn.setAttribute(Qt.WA_TranslucentBackground)
        self.okBtn.clicked.connect(self.okBtnClick)
        self.okBtn.setStyleSheet(
            "color: #ffffff;border: 0px ;background:rgba(0,0,0,0)")
        self.okBtn.hide()

    def organicBtnClick(self):
        self.label.setStyleSheet("background: white; "
                                 "background-position: center;"
                                 "background-repeat: no-repeat;"
                                 "background-image: url(greenbin.png) ;"
                                 "border-radius:25px;")

        self.inorganicBtn.hide()
        self.organicBtn.hide()
        self.label2.setText("Put your waste in Green bin")
        self.okBtn.show()
        self.flag=0


    def inorganicBtnClick(self):
        self.label.setStyleSheet("background: white; "
                                 "background-position: center;"
                                 "background-repeat: no-repeat;"
                                 "background-image: url(bluebin.png) ;"
                                 "border-radius:25px;")

        self.inorganicBtn.hide()
        self.organicBtn.hide()
        self.label2.setText("Put your waste in Blue bin")
        self.okBtn.show()
        self.flag=1

# calls the prediction function on click 
    def okBtnClick(self):
        print("reading img and classifying")
        model=tf.keras.models.load_model("waste_segregation.model") #loading of trained model
        DIVISION=["Non-Biodegradable","Biodegradable"]
        SIZE=70 # size we needed for processed images

        def process(path):   #function for processing the test image
            img_array=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img_array_new=cv2.resize(img_array,(SIZE,SIZE))
            n = (img_array_new.reshape(-1,SIZE,SIZE,1))/255
            return n

        root="A:\BITS\AOS project\image classifier\Test1"
        for img in (os.listdir(root)):
            pred=model.predict([process(os.path.join(root,img))])

            if(pred <= 0.7): #adjusting sigmoid since images in recyclable dataset is approx(300) less
                print("Non-Biodegradable")
                self.obt_flag=1
            else:
                print("Biodegradable")
                self.obt_flag=0
   
        if (self.obt_flag != self.flag):
            #change image to sad smiley
            self.label.setStyleSheet("background: white; "
                                 "background-position: center;"
                                 "background-repeat: no-repeat;"
                                 "background-image: url(sad.png) ;"
                                 "border-radius:25px;")
            self.label2.setText("You have thrown your waste in wrong bin")
            self.okBtn.hide()
            
        else:
            #successfully disposed 
            self.label.setStyleSheet("background: white; "
                                 "background-position: center;"
                                 "background-repeat: no-repeat;"
                                 "background-image: url(smiley.png) ;"
                                 "border-radius:25px;")
            self.label2.setText("Disposed your waste correctly!!")
            self.okBtn.hide()
                
def main():
    App = QApplication(sys.argv)

    window = Window(900, 600) #size of the display window  
    sys.exit(App.exec())


if __name__ == '__main__':
    main()
