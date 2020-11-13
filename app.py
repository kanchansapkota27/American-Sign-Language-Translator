import numpy as np
import cv2,sys,csv,os,threading
from PyQt5.QtWidgets import QMessageBox,QAction
from keras.models import load_model
import qimage2ndarray as q2a
from PyQt5 import QtWidgets,QtGui,QtCore,QtCore
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication,QMainWindow,QMessageBox,QFileDialog,QInputDialog
from PyQt5.QtGui import QImage, QPixmap
import pyttsx3
from ui import Ui_MainWindow
import constants
from PyQt5.QtCore import QThread,pyqtSignal,pyqtSlot,QObject


class MainWindow(QMainWindow):
    speech_seq_signal=pyqtSignal(str)
    def __init__(self):
        super(MainWindow,self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.showMaximized()
    #Menu and Action
        about_action=QAction('About',self)
        about_action.triggered.connect(self.show_about)
        self.ui.menuOptions.addAction(about_action)
    #Initalize Buttons
        self.ui.speak_btn.clicked.connect(self.start_speak_thread)
        self.ui.speak_btn.setShortcut("S")
        self.ui.speak_btn.setToolTip("Shortcut: S")
    #Initilizing paremeters
        self.ui.outTextEdit.setReadOnly(True)
    #ComboBox
        self.others=['All','Model Loss','Model Accuracy']
        signNames=[files.split('.')[0] for files in os.listdir(constants.signImagePath)]
        self.ui.search_drop_comboBox.addItems(signNames)
        self.ui.search_drop_comboBox.insertSeparator(len(signNames))
        self.ui.search_drop_comboBox.addItems(self.others)
        self.ui.search_drop_comboBox.insertSeparator((len(signNames)+2))
    #Buttons
        self.ui.search_sign_btn.clicked.connect(self.displaySign)
    #Speech Worker
        self.speak_seq=''
        self.speech_worker=SpeechWorker(self.speak_seq)
        self.speech_seq_signal.connect(self.speech_worker.get_sequnece)

    #Threading Stuffs
        self.video_worker=VideoWorker(self)
        self.video_worker.image_signals.connect(self.show_pixmap)
        self.video_worker.pred_text_signal.connect(self.update_Texts)
        self.video_worker.start()

    def show_pixmap(self,image:list):
        frame,thresh =image
        w_f,h_f=self.ui.video_label.width(),self.ui.video_label.height()
        w_f,h_f=w_f-200,h_f-100
        frame=cv2.resize(frame,(w_f,h_f),cv2.INTER_AREA)
        frame_img=q2a.array2qimage(frame)
        frame_img = frame_img.rgbSwapped()
        self.ui.video_label.setPixmap(QPixmap.fromImage(frame_img))

        w_t,h_t=self.ui.thresh_video_label.width(),self.ui.thresh_video_label.height()
        w_t,h_t=w_t-100,h_t-100
        thresh=cv2.resize(thresh,(w_t,h_t),cv2.INTER_AREA)
        thresh_img=q2a.array2qimage(thresh)
        thresh_img = thresh_img.rgbSwapped()
        self.ui.thresh_video_label.setPixmap(QPixmap.fromImage(thresh_img))

    def update_Texts(self,letter,sequence):
        self.speak_seq=sequence
        self.ui.current_letter_label.setText(letter)
        self.ui.outTextEdit.clear()
        self.ui.outTextEdit.append(sequence)


    def displaySign(self):
        selectedSign=self.ui.search_drop_comboBox.currentText()
        signPath=os.path.join(constants.signImagePath,f'{selectedSign}.png')
        if selectedSign in self.others:
            signPath=os.path.join(constants.othersPath,f'{selectedSign}.png')
        pixmap=QPixmap(signPath)
        self.ui.sign_image_label.setPixmap(pixmap)
        self.ui.sign_image_label.setScaledContents(True)


    def show_about(self):
        message="Developed By:\n Kanchan Sapkota\nSys ID:2017009931\nRoll No:170101113\nB.Tech CSE VII'A'"
        QMessageBox.about(self,"About",message)

    def start_speak_thread(self):
        self.speech_worker.start()
        print(self.speak_seq)
        self.speech_seq_signal.emit(self.speak_seq)





class VideoWorker(QThread):
    image_signals=pyqtSignal(list)
    pred_text_signal=pyqtSignal(str,str)
    def __init__(self,parent=None):
        QThread.__init__(self,parent)
        self.running=False
    def run(self):
        self.model=load_model(constants.modelPath)
        self.is_predict_roi=True
        cap=cv2.VideoCapture(0)
        sequence=constants.start_sequence
        pred_text=''
        pred_letter=''
        del_count=0
        prev_pred=None
        consecutive_goal=constants.consecutive_goal
        consecutive_count=0
        while True:
            ret,frame=cap.read()
            if ret:
                frame=cv2.flip(frame,1)
                y,x,_=frame.shape
                x1,y1=(x//2+50,0)
                x2,y2=(x,y//2+50)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                crop_img = frame[y1:y2, x1:x2]
                blur = cv2.GaussianBlur(crop_img,(7,7),0)
                grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((5,5))
                dilation = cv2.dilate(grey,kernel,iterations=1)
                erosion = cv2.erode(dilation,kernel,iterations=1)
                blur1 = cv2.GaussianBlur(erosion,(5,5),0)
                thresh = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
                if self.is_predict_roi:
                    img = cv2.resize(thresh, (50,50))
                    img = img.reshape(1,50,50,1)
                    img = img/255.0
                    prd = self.model.predict(img)
                    index = prd.argmax()
                    pred_text=constants.gestures[index]
                    pred_letter=pred_text
                    if prev_pred==pred_text:
                        consecutive_count+=1
                    else:
                        consecutive_count=0
                    if pred_text=='BG':
                        pred_text=''
                    if pred_text=="SPACE":
                        del_count+=1
                        pred_text=""
                        if del_count>=consecutive_goal:
                            sequence+=" "
                            del_count=0

                    if pred_text=="DEL":
                        del_count+=1
                        pred_text=''
                        if del_count>consecutive_goal:
                            sequence=sequence[:-1]
                            del_count=0
                    if consecutive_count>consecutive_goal:
                        sequence+=pred_text
                        consecutive_count=0
                images=[frame,thresh]
                self.image_signals.emit(images)
                self.pred_text_signal.emit(pred_letter,sequence)
                prev_pred=pred_text

class SpeechWorker(QThread):
    def __init__(self,sequence,parent=None):
        QThread.__init__(self,parent)
        self.sequence=''
        self.running=False

    def run(self):
        engine=pyttsx3.init()
        engine.setProperty('rate',constants.voiceSpeed)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[constants.voiceId].id)
        engine.say(self.sequence)
        engine.runAndWait()

    @pyqtSlot(str)
    def get_sequnece(self,sequence):
        self.sequence=sequence












if __name__ == "__main__":
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())


