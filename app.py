##### Required Imports
from tkinter import *
from tkinter import messagebox as msgbx
from tkinter import filedialog as fd
from datetime import datetime
import os
import shutil
import re
import random
import time
import smtplib
import csv
from urllib3 import PoolManager
from email.message import EmailMessage
from credentials import EMAIL_ADDR, PASSWORD # Email and Password stored in credentials.py
from PIL import Image, ImageTk # pip install pillow
import cv2 # pip install opencv-python
import numpy as np # pip install numpy
import dlib # pip install dlib
import pytesseract as pytrt # pip install pytesseract
import face_recognition as fr # pip install face-recognition
import speech_recognition as sr # pip install SpeechRecognition
from playsound import playsound # pip install playsound
from gtts import gTTS # pip install gTTS
from googletrans import Translator # pip install googletrans
from pynput.mouse import Listener # pip install pynput
import pyautogui as pyg # pip install PyAutoGUI
import qrcode # pip install qrcode

##### Utility Functions
def is_connected():
    ''' Checks wheather an active connection is present or not '''
    try:
        http = PoolManager()
        r = http.request('GET', 'https://www.google.co.in')
        return r.status == 200
    except Exception:
        return False

def speech_recognizer():
    ''' Listen to Outside Sounds and
    convert them to text '''
    tts = ""
    r = sr.Recognizer()
    print("Speak...")
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        tts = r.recognize_google(audio)
    except:
        print("Not Recognized")
    # print(text)
    return tts

def get_language_code(text):
    ''' Fetches Language Code of the input text '''
    trans = Translator()
    lang_code = trans.detect(text).lang

    return lang_code

def capture_img(output_path, img_name):
    ''' Capture a photo using webcam '''
    cap = cv2.VideoCapture(0)
    i = 0
    while i<=10:
        _, frame = cap.read()
        cv2.imwrite(output_path+img_name, frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

def send_email(to_mail, path_to_img):
    ''' Send Alert Mail '''
    msg = EmailMessage()
    msg['Subject'] = "Ghost Sense"
    msg['From'] = EMAIL_ADDR
    msg['To'] = to_mail
    msg.set_content('Intruder Alert!')

    with open(path_to_img, 'rb') as f:
        data = f.read()
        file_name, file_extension = os.path.splitext(path_to_img)
        file_extension = file_extension.strip('.') # Remove the preceding '.' from extension
        file_name = file_name[13:]

    msg.add_attachment(data, maintype='image', subtype=file_extension, filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:

        smtp.login(EMAIL_ADDR, PASSWORD)
        smtp.send_message(msg)

def generate_qrcode(data, output_id):
    ''' Function responsible for generating
    qrcodes of data provided '''
    qr = qrcode.QRCode(
    version=None,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(f"./img_txt_output/{output_id}.png")

def create_folder(folder_name):
    ''' Create Folder if does not exists '''
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

def calculate_EAR(eye):
    ''' Function responsible for calculating
    Eye Aspect Ratio (EAR) '''
    print(eye[1])
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    eye_aspect_ratio = (A+B)/(2.0*C)
    return eye_aspect_ratio

######### GUI Configurations
root = Tk()
root.geometry("950x700")
root.resizable(False, False)
root.title("System Sense")

app_icon = PhotoImage(file="./assets/icons/app_icon.png")
root.iconphoto(True, app_icon)

##### Variables
email = StringVar()
press_count = 0

##### Button Commands
def start_recording():
    '''Function responsible for WebCam/Screen Record '''
    choice = msgbx.askquestion("Record", "Would you like to Face Cam?")
    if choice == "yes":
        msg = "Recording will begin within 3 seconds. Press 'Esc' to stop."
        msgbx.showinfo("Record", msg)
        time.sleep(3)

        cap = cv2.VideoCapture(0)
        cap.set(3, 600) # Set Width
        cap.set(4, 500) # Set Height

        FW = int(cap.get(3)) # Get Frame Width
        FH = int(cap.get(4)) # Get Frame Height
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        if not os.path.isdir("./output_video/"): # Make output dir if not present already
            os.mkdir("output_video")
        OUTPUT_DIR = "./output_video/"
        FILE_NAME = datetime.now().strftime("%A %m-%d-%Y, %H:%M:%S") + ".mp4" # Save with the current date's name

        four_cc = cv2.VideoWriter_fourcc(*'FMP4')
        output = cv2.VideoWriter(OUTPUT_DIR + FILE_NAME, four_cc, 8.5, (FW, FH)) # Ouput Recording File

        while cap.isOpened():
            rtn, frame = cap.read()
            date = datetime.now().strftime("%A %m/%d/%Y, %H:%M:%S")

            if rtn:
                frame = cv2.putText(frame, date, (18, 30), FONT, 0.8, (255, 255, 255), 2)
                output.write(frame)
                cv2.imshow('Record', frame)

                if (cv2.waitKey(1) & 0xFF) == 27: # 'Esc' key to stop recording
                    break
            else:
                msgbx.showerror("Error!", "Could Not Read from Webcam :(")
                break

        cap.release()
        output.release()
        cv2.destroyAllWindows()
    
    else:
        choice = msgbx.askquestion("Record", "Would you like to Screen record?")
        if choice == "yes":
            if not os.path.isdir("./output_video/"): # Make output dir if not present already
                os.mkdir("output_video")
            
            SCREEN_WIDTH = root.winfo_getscreenwidth()
            SCREEN_HEIGHT = root.winfo_getscreenheight()
            RESOLUTION, FPS = ((SCREEN_WIDTH,SCREEN_HEIGHT), 8.5)
            OUTPUT_DIR = "./output_video/"
            FILE_NAME = "ScreenRecording " + datetime.now().strftime("%m-%d-%Y, %H-%M-%S") + ".mp4"

            four_cc = cv2.VideoWriter_fourcc(*'FMP4')
            output = cv2.VideoWriter(OUTPUT_DIR + FILE_NAME, four_cc, FPS, RESOLUTION) 

            # Live Screen Recording Previe
            cv2.namedWindow("Record", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("Record", 500, 400)

            while True:
                img = pyg.screenshot() 
                img = np.array(img)

                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                output.write(frame)
                cv2.imshow("Record", frame)

                if cv2.waitKey(1) & 0xFF == 27: # Escape to Stop
                    break
            
            output.release()
            cv2.destroyAllWindows()
        else:
            msgbx.showerror("Record", "No option selected!")

def sound_sense():
    ''' Function responsible for handling noise/sound events '''
    if not is_connected():
        msg = "Your system is not online. This feature requires internet!"
        msgbx.showinfo("Connection Error", msg)

    else:
        if not os.path.isdir("./output_audio/"): # Make output dir if not present already
            os.mkdir("output_audio")
        OUTPUT_DIR = "./output_audio/"
        FILE_NAME = datetime.now().strftime("%A %m-%d-%Y, %H:%M:%S") + ".mp3"
        msgbx.showinfo("Sound Sense", "Sound Sense Activated!")

        new_window = Toplevel()

        new_window.geometry("405x400")
        new_window.minsize(405, 400)
        new_window.maxsize(405, 400)
        new_window.title("Sound Sense")
        new_window.config(bg="black")

        Label(new_window, text="Processing...", bg="black",
             fg="red", font="copperplate 30 bold").place(relx=0.28, rely=0.08)

        try:
            frames = [PhotoImage(file='./assets/images/sound_sense.gif',
                        format=f'gif -index {j}') for i in range(2) for j in range(10)]
            def update(index):
                ''' Show Animations '''
                try:
                    frame = frames[index]
                    index += 1
                    label.configure(image=frame)
                    new_window.after(100, update, index)
                except:
                    msgbx.showinfo("Sound Sense", "Scanning Surroundings!\nSpeak 'Deactivate' to Stop.")
                    output = ""

                    while len(output)<=30:
                        text = speech_recognizer()
                        output += (text + ".")

                        if text.lower() == "deactivate":
                            playsound("./assets/bg_music/exit_msg.mp3")
                            new_window.destroy()
                            break
                    else:
                        lang_code = get_language_code(output)
                        print(lang_code)
                        playsound("./assets/bg_music/beep.mp3")
                        to_voice = gTTS(text=output, lang=lang_code, slow=False)
                        to_voice.save(OUTPUT_DIR + FILE_NAME)
                        new_window.destroy()
                        msgbx.showwarning("Sound Sense", "Sound Waves Threshold Limit Exceeded!")

            label = Label(new_window, bg="black", relief=SUNKEN, bd=4)
            label.place(rely=0.25)
            new_window.after(0, update, 0)
        except:
            pass

def motion_detect():
    ''' Detect Motion Inside Frame'''
    count = 0 # Counter
    msg = "Initiating Motion Detect. Press 'Esc' to stop."
    msgbx.showinfo("Motion Sense", msg)
    time.sleep(3)

    if not os.path.isdir("./output_video/"): # Make output dir if not present already
        os.mkdir("output_video")
    OUTPUT_DIR = "./output_video/"
    FILE_NAME = datetime.now().strftime("%A %m-%d-%Y, %H:%M:%S") + ".mp4" # Save with the current date's name

    cap = cv2.VideoCapture(0)
    FW = int(cap.get(3)) # Get Frame Width
    FH = int(cap.get(4)) # Get Frame Height

    four_cc = cv2.VideoWriter_fourcc(*'FMP4')
    output = cv2.VideoWriter(OUTPUT_DIR + FILE_NAME, four_cc, 8.5, (FW, FH)) # Ouput Recording File

    while cap.isOpened():
        rtn1, frame1 = cap.read()
        rtn2, frame2 = cap.read()

        if rtn1 and rtn2:
            diff = cv2.absdiff(frame2, frame1)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(diff, (5,5), 0)
            _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)

            contr, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contr) > 25:
                max_cnt = max(contr, key=cv2.contourArea)
                x,y,w,h = cv2.boundingRect(max_cnt)
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(frame1, "MOTION DETECTED", (18,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                count += 0.5
                # print(count)
            if count>=35:
                playsound("./assets/bg_music/beep.mp3")
                msgbx.showwarning("Motion Sense", "Motion Activity Level Exceeded!")
                break

            date = datetime.now().strftime("%A %m/%d/%Y, %H:%M:%S")
            cv2.putText(frame1, date, (13,690), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            output.write(frame1)
            cv2.imshow("Motion Sense", frame1)

            if cv2.waitKey(1) & 0xFF == 27: # Escape to Stop
                break
        else:
            msgbx.showerror("Error!", "Could Not Read from Webcam :(")
            break

    cv2.destroyAllWindows()
    output.release()
    cap.release()

def ghost_sense():
    ''' Detects Mouse/ Keyboard Behaviors '''
    if not is_connected():
        msg = "Your system is not online. This feature requires internet!"
        msgbx.showinfo("Connection", msg)
        return
        
    regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    if not re.search(regex, email.get()):
        msgbx.showerror("Ghost Sense", "Receiver's mail address not found!")
        return

    else:
        if not os.path.isdir("./output_img/"): # Make output dir if not present already
            os.mkdir("output_img")
        OUTPUT_DIR = "./output_img/"
        FILE_NAME = datetime.now().strftime("%A %m-%d-%Y_%H:%M:%S") + ".png"
        msgbx.showinfo("Ghost Sense", "Ghost Sense Activated!")

        def on_click(x, y, button, pressed):
            ''' Handles Mouse Events and
            increase the pointer click count '''
            global  press_count
            if pressed:
                press_count += 1

            if press_count > 5:
                listener.stop()
        # Collect mouse events until released
        with Listener(on_click=on_click) as listener:
            listener.join()

        if press_count > 5:
            playsound("./assets/bg_music/pred_sense.mp3")
            capture_img(OUTPUT_DIR, FILE_NAME)
            send_email(email.get(), OUTPUT_DIR+FILE_NAME)
            print("Sent!")
            msgbx.showwarning("Ghost Sense", "Unauthorized Access Detected!")

def recognize_face():
    ''' Compares Unauthorised Faces with
    the user's face '''
    if not os.path.isdir("./output_img/"): # Make output dir if not present already
        os.mkdir("output_img")
    
    if not os.path.isdir("./face_to_recognize/"): # Make output dir if not present already
        os.mkdir("face_to_recognize")

    if not os.listdir("./face_to_recognize/"):
        response = msgbx.askquestion("Add Face Now", "No registered faces found! Do you want to add a face now?")
        if response == "no":
            msgbx.showerror("Face Detect", "Kindly place a face to be detected!")
            return
        else:
            msgbx.showinfo("Face Detect", "Stay Still, when ready press 'Space' to initialize Face Registration")
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                rtn, frame = cap.read()

                if rtn:
                    cv2.imshow("Face Register", frame)
                    cv2.imwrite("./face_to_recognize/registered_face.jpg", frame)

                    if cv2.waitKey(1) & 0xFF == 32:
                        break
                else:
                    msgbx.showerror("Error!", "Could Not Read from Webcam :(")
                    break

            cap.release()
            cv2.destroyAllWindows()
            msgbx.showinfo("Face Detect", "Face Successfully Registered!")

    else:
        if os.path.exists("./face_to_recognize/.DS_Store"):
            os.remove("./face_to_recognize/.DS_Store")
        user_img_name = os.listdir("./face_to_recognize/")[0] # The face to be recognized
        print(user_img_name)
        msgbx.showinfo("Face Detect", "Do not Forget to put your face image in the correct folder")
        time.sleep(0.5)
        msgbx.showinfo("Face Detect", "Stay Still, when ready press 'Space' to initialize Face Detect")
        recent_img = ""

        rem_attempts = 3
        face_cascade = cv2.CascadeClassifier("./assets/haarcascade_frontalface_default.xml")
        while rem_attempts > 0:
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                rtn1, frame = cap.read()
                rtn2, frame2 = cap.read()

                if rtn1 and rtn2:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Press 'Space' when ready", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.8, (255, 255, 255), 2)
                    
                    cv2.imshow("Face Detect", frame)
                    img_name = datetime.now().strftime("%A %m-%d-%Y, %H:%M:%S") + ".jpg"
                    recent_img = img_name

                    if cv2.waitKey(1) & 0xFF == 32: # Space Bar
                        cv2.imwrite(f"./output_img/{img_name}", frame2)
                        break

                else:
                    msgbx.showerror("Error!", "Could Not Read from Webcam :(")
                    break

            cap.release()
            cv2.destroyAllWindows()

            face_user = fr.load_image_file(f"./face_to_recognize/{user_img_name}")
            face_user = cv2.cvtColor(face_user, cv2.COLOR_BGR2RGB)
            print(recent_img)
            captured_img = fr.load_image_file(f"./output_img/{recent_img}")
            captured_img = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
            
            try:
                user_face_encode = fr.face_encodings(face_user)[0]
                captured_img_encode = fr.face_encodings(captured_img)[0]
            except:
                playsound("./assets/bg_music/beep.mp3")
                rem_attempts -= 1
                if rem_attempts == 0:
                    msgbx.showerror("Face Detect", f"Access Denied!")
                else:
                    msgbx.showerror("Face Detect", f"Access Denied! Could not recognize face.\
                                    \n\nAttempt(s) Remaining: {rem_attempts}")
            else:
                is_match = fr.compare_faces([user_face_encode], captured_img_encode)[0]
                # print(is_match)
                if is_match:
                    playsound("./assets/bg_music/access_granted.mp3")
                    msgbx.showinfo("Face Detect", "Access Granted!")
                    break
                else:
                    playsound("./assets/bg_music/beep.mp3")
                    rem_attempts -= 1
                    if rem_attempts == 0:
                        msgbx.showerror("Face Detect", f"Access Denied!")
                    else:
                        msgbx.showerror("Face Detect", f"Access Denied! Could not recognize face.\
                                    \n\nAttempt(s) Remaining: {rem_attempts}")

def detect_drowsiness():
    ''' Detects drowsiness using 
    Eye Aspect Ratio '''
    msgbx.showinfo("DrowZzz Detect", "Initiating DrowZzz Detect!")
    cap = cv2.VideoCapture(0)
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("./assets/shape_predictor_68_face_landmarks.dat")
    alarm_counter = 0

    while True:
        rtn, frame = cap.read()

        if rtn:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = hog_face_detector(gray)
            for face in faces:
                face_landmarks = dlib_facelandmark(gray, face)
                left_eye = []
                right_eye = []

                for n in range(36,42): # For right eye detection
                    x1 = face_landmarks.part(n).x
                    y1 = face_landmarks.part(n).y
                    left_eye.append((x1,y1))
                    next_point = n+1
                    if n == 41:
                        next_point = 36
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),1)

                for n in range(42,48): # For left eye detection
                    x1 = face_landmarks.part(n).x
                    y1 = face_landmarks.part(n).y
                    right_eye.append((x1,y1))
                    next_point = n+1
                    if n == 47:
                        next_point = 42
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),1)

                left_EAR = calculate_EAR(left_eye)
                right_EAR = calculate_EAR(right_eye)

                final_EAR = (left_EAR+right_EAR)/2
                final_EAR = round(final_EAR,2)
                if final_EAR < 0.15:
                    alarm_counter += 1
                    print("Drowsy")
                    print(final_EAR)
            
            cv2.imshow("DrowZzz", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if alarm_counter >= 5:
                playsound("./assets/bg_music/alarm.mp3")
                msgbx.showinfo("DrowZzz Detect", "You are feeling DrowZZZ, Why not take a break?")
                break

        else:
            msgbx.showerror("Error!", "Could Not Read from Webcam :(")
            break

    cap.release()
    cv2.destroyAllWindows()

def get_text():
    ''' Extract text from image '''
    if not os.path.isdir("./img_txt_output/"):
        os.mkdir("img_txt_output")
    file = fd.askopenfile("r", filetypes=[("JPEG", ".jpg"), ("PNG", ".png")])

    if file:
        img_path = file.name
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        blurred_img = cv2.GaussianBlur(gray, (5,5), 0) # Blur Img to reduce noise

        ## Convert img to Black and White using Adaptive Threshold
        img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 85, 11)
        text = pytrt.image_to_string(img)
        img_id = id(text)
        to_write = {
            'Image Name': os.path.split(file.name)[-1],
            "Image Id": img_id,
            'Extension': file.name.split(".")[-1],
            'Created On': datetime.now().strftime("%a %m-%d-%Y"),
            'Extracted Text': text
        }
        with open("./img_txt_output/fetched_data.csv", "a") as csvfile: # Write the output to a csv file
            dict_writer = csv.DictWriter(csvfile, fieldnames=to_write.keys())
            dict_writer.writeheader()
            dict_writer.writerow(to_write)
            
        generate_qrcode(text, str(img_id))
        msgbx.showinfo("Img-Text", "Text Extraction Successful!")
    else:
        msgbx.showerror("Img-Text", "Error Loading File!")

def organize_dir():
    ''' Organise the files inside a
    directory '''
    exts = {
        'Images' : ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.ico', '.svg', '.tff'],
        'Medias' : ['.mp4', '.mp3', '.mkv', '.mov', '.3gp', '.wav', '.wmv', '.aac', '.flv', '.webm'],
        'Docs' : ['.pdf', '.docx', '.doc', '.pptx', '.pages', '.key', '.txt', '.rtf', '.csv', '.xlsx', '.odt', '.ppt'],
        'Codes' : ['.py', '.pyc', '.dart', '.c', '.cpp', '.js', '.html', '.css', '.java', '.go', '.r', '.sh'],
        'Archives' : ['.rar', '.zip', '.7z', '.alzip', '.iso', '.dmg'],
        'Executables' : ['.exe', '.bat', '.command', '.apk', '.app']
    }
    msgbx.showinfo("OrganiZ", "Choose a Directory which you would like to Organize...")
    time.sleep(0.5)
    dir_path = fd.askdirectory()

    if dir_path: 
        os.chdir(dir_path)
        cwd_files = [file.lower() for file in os.listdir(dir_path)]
        print(cwd_files)
        # exit()
        file_count = len(next(os.walk(dir_path))[2])

        for file in cwd_files:
            if os.path.isfile(file):
                ext = os.path.splitext(file) # Split the file into its name and extension

                if (ext[1] in exts.get('Images')):
                    create_folder('Images')
                    shutil.move(file, './Images/')
                
                elif (ext[1] in exts.get('Medias')):
                    create_folder('Media')
                    shutil.move(file, './Media/')
                
                elif (ext[1] in exts.get('Docs')):
                    create_folder('Docs')
                    shutil.move(file, './Docs/')
                
                elif (ext[1] in exts.get('Codes')):
                    create_folder('Codes')
                    shutil.move(file, './Codes/')
                
                elif (ext[1] in exts.get('Archives')):
                    create_folder('Archives')
                    shutil.move(file, './Archives/')
                
                elif (ext[1] in exts.get('Executables')):
                    create_folder('Exec')
                    shutil.move(file, './Exec/')
                
                else:
                    create_folder('Others')
                    shutil.move(file, './Others/')

        msgbx.showinfo("SUCCESS!", f"Your Directory Has Been Cleaned :)\n{file_count} file(s) Have Been Cleaned")

################################ Main Interface #################################
##### Displaying Background Image
img =  Image.open("./assets/images/bg.jpg")
bg_img = ImageTk.PhotoImage(img)

bg_img_label = Label(root,image=bg_img)
bg_img_label.place(relwidth=1, relheight=1)

##### Header
header_label = Label(root, text="System Sense", font="copperplate 40 bold", fg="red",
                    bg="black", relief=GROOVE, bd=3, padx=3, pady=3)
header_label.place(relx=0.37, rely=0.05)

sub_header_label = Label(root, text="Advanced Side Of Your System", font="copperplate 25 bold", fg="#d69f09",
                        bg="black", relief=GROOVE, bd=3, padx=3, pady=3)
sub_header_label.place(relx=0.31, rely=0.15)

##### Button Icons
face_icon = PhotoImage(file="./assets/icons/face_recognition.png")
motion = PhotoImage(file="./assets/icons/motion.png")
g_sense_icon = PhotoImage(file="./assets/icons/ghost.png")
sound = PhotoImage(file="./assets/icons/sound.png")
drowsiness_icon = PhotoImage(file="./assets/icons/drowsy.png")
record = PhotoImage(file="./assets/icons/record.png")
img_to_txt = PhotoImage(file="./assets/icons/img_txt.png")
organizer_icon = PhotoImage(file="./assets/icons/organize.png")
exit_icon = PhotoImage(file="./assets/icons/exit.png")

##### Buttons
### Upper Row Buttons
motion_btn = Button(root, text="M-Sense", image=motion, compound=TOP,
                    font="copperplate 25", command=motion_detect)
motion_btn.config(fg="#bd557f", highlightbackground="#6681c4", 
                  highlightthickness=5, cursor="hand")
motion_btn.place(relx=0.08, rely=0.30)

sound_btn = Button(root, text="S-Sense", image=sound, compound=TOP,
                  font="copperplate 25", command=sound_sense)
sound_btn.config(fg="seagreen", highlightbackground="#b09d33", 
                highlightthickness=5, cursor="hand")
sound_btn.place(relx=0.31, rely=0.30)

face_rcg_btn = Button(root, text="LOCK-IT", image=face_icon, compound=TOP,
                    font="copperplate 25")
face_rcg_btn.config(fg="#2235bf", highlightbackground="#4fbd24", highlightthickness=5,
                    cursor="hand", command=recognize_face)
face_rcg_btn.place(relx=0.545, rely=0.30)

g_sense_btn = Button(root, text="G-Sense", image=g_sense_icon, compound=TOP,
                       font="copperplate 25", command=ghost_sense)
g_sense_btn.config(fg="black", highlightbackground="#db5151", 
                     highlightthickness=5, cursor="hand")
g_sense_btn.place(relx=0.775, rely=0.30)

### Lower Row Buttons
record_btn = Button(root, text="Record", image=record, compound=TOP,
                    font="copperplate 22", command=start_recording)
record_btn.config(fg="#b50940", highlightbackground="#914fb3", 
                 highlightthickness=3, cursor="hand")
record_btn.place(relx=0.18, rely=0.64)

img_txt_btn = Button(root, text="Img-Txt", image=img_to_txt, compound=TOP,
                    font="copperplate 22", command=get_text)
img_txt_btn.config(fg="#de5454", highlightbackground="#2db393", 
                 highlightthickness=3, cursor="hand")
img_txt_btn.place(relx=0.35, rely=0.64)

organize = Button(root, text="Organiz", image=organizer_icon, compound=TOP,
                    font="copperplate 21")
organize.config(fg="#2577db", highlightbackground="#d14f92", highlightthickness=3,
                    cursor="hand", command=organize_dir)
organize.place(relx=0.52, rely=0.64)

detect_drowsiness_btn = Button(root, text="Drowzz", image=drowsiness_icon, compound=TOP,
                    font="copperplate 22")
detect_drowsiness_btn.config(fg="#cf2357", highlightbackground="#2b58ba", highlightthickness=3,
                    cursor="hand", command=detect_drowsiness)
detect_drowsiness_btn.place(relx=0.70, rely=0.64)

exit_btn = Button(root, image=exit_icon, font="copperplate 22", 
                 padx=5, pady=5, command=root.quit)
exit_btn.config(highlightbackground="black", highlightthickness=3, cursor="hand")
exit_btn.place(relx=0.94, rely=0.91)

##### Email Widget
frame = Frame(root, width=580, height=50, bg="black")

mail_label = Label(frame, text="Send Alert At", font="copperplate 25",
                  fg="red", bg="black")
mail_label.place(rely=0.1)

email_entry = Entry(frame, textvariable=email, width=32, font="comicsansms 17",
                    bg="black", fg="wheat", relief=GROOVE, bd=3)
email_entry.config(highlightthickness=3, highlightbackground='black')
email_entry.place(relx=0.35, rely=0.13)

frame.place(relx=0.01, rely=0.9)

root.mainloop()


