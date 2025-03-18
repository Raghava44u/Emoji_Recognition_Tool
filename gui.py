import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

emotion_model = load_model("E:/projects/Emoji_Reco/emotion_model.keras")

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emoji_dist = {
    0: "./emojis/angry.png",
    1: "./emojis/disgusted.png",
    2: "./emojis/fearful.png",
    3: "./emojis/happy.png",
    4: "./emojis/neutral.png",
    5: "./emojis/sad.png",
    6: "./emojis/surprised.png"
}

cap1 = cv2.VideoCapture(0)
if not cap1.isOpened():
    print("Cannot open camera")
    exit()

global last_frame1, current_emotion
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
current_emotion = -1 

def show_vid():
    global current_emotion
    flag1, frame1 = cap1.read()
    if not flag1:
        print("Error capturing video frame")
        return
    
    frame1 = cv2.flip(frame1, 1) 
    frame1 = cv2.resize(frame1, (600, 500))
    
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray_frame[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0 
        
        prediction = emotion_model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(prediction))

        if maxindex != current_emotion: 
            current_emotion = maxindex
            show_vid2() 

        print(f"Predicted Emotion: {emotion_dict[maxindex]}, Confidence: {prediction[0][maxindex]:.2f}")

    global last_frame1
    last_frame1 = frame1.copy()
    pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_vid)

def show_vid2():
    if current_emotion != -1:
        try:
            frame2 = cv2.imread(emoji_dist[current_emotion])
            frame2 = cv2.resize(frame2, (250, 250))
            pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(pic2)
            imgtk2 = ImageTk.PhotoImage(image=img2)
            lmain2.imgtk2 = imgtk2
            lmain2.configure(image=imgtk2)
            lmain3.configure(text=emotion_dict[current_emotion], font=('arial', 45, 'bold'))
        except Exception as e:
            print(f"Error displaying emoji: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'

    img = ImageTk.PhotoImage(Image.open("logo2.jpg"))
    heading = Label(root, image=img, bg='black')
    heading.pack()
    heading2 = Label(root, text="Photo to Emoji", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')
    heading2.pack()

    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)
    
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold'))
    exitbutton.pack(side=BOTTOM)
    
    show_vid()
    root.mainloop()

    cap1.release()
    cv2.destroyAllWindows()