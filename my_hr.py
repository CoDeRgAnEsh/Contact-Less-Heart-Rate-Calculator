from os import system
import cv2 as cv
import numpy
from matplotlib import pyplot as plt
from math import sqrt, log, exp, pi
import time
from amplify_spatial_Gdown_temporal_ideal import amplify_spatial_Gdown_temporal_ideal
import matplotlib.animation as animation
import threading
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")	
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import time
import sys


frame = numpy.zeros((640,480,3),numpy.uint8)
im = numpy.zeros((640,480,3),numpy.uint8)
# beginning=0
# end=0
roi_frames=[]
magnified_frames=[]


def record():
	global frame , frames
	frames=[]
	periods=[]
	count=0
	capture = cv.VideoCapture(0)
	

	while(True):
		# print(time.time())
		start = time.time()
		ret,fr = capture.read()
		# gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		# cv.imshow("Frame",gray)
		# if cv.waitKey(1) & 0xFF == ord('q'):
		# 	break
		count+=1
		end=time.time()

		if(ret==False):
			continue
		# else:
		# 	cv.imshow("Frame",frame)

		frames.append(fr)
		frame=frames.pop()
		# print("frame" + str(time.time()))
		periods.append(end-start)

	# print(periods)
	# cap.release()
	# cv.destroyAllWindows()

# 	cap = cv.VideoCapture(0)

# 	while(True):
#     # Capture frame-by-frame
# 	    ret, frame = cap.read()

# 	    # Our operations on the frame come here
# 	    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# 	    # Display the resulting frame
# 	    cv.imshow('frame',frame)
# 	   

# # When everything done, release the capture
# 	cap.release()
# 	cv2.destroyAllWindows()

def magnify_video(beginning,end):
	global roi_frames,magnified_frames
	target_frames=roi_frames[beginning:end]
	fourcc = cv.VideoWriter_fourcc(*'MP4V')
	mag = cv.VideoWriter('output.mp4',fourcc, 30, (120,96))
    
	for b, fr in enumerate(target_frames):
		fr = cv.resize(fr, (120,96), interpolation = cv.INTER_AREA)
		mag.write(fr)

	mag.release()

	amplify_spatial_Gdown_temporal_ideal("output.mp4","", 80,4,30.0/60.0,160.0/60.0,1.0,'rgb')

	vid = cv.VideoCapture("proc.mov")
	while(True):
		check,output=vid.read()
		if(check==True):
			magnified_frames.append(output)
		else:
			break
	vid.release()
	# print(str(beginning) + " " + str(time.time()))
	# print(len(roi_frames))

def start_func():
	global frame,frames,roi_frames,magnified_frames
	rec = threading.Thread(target=record)
	rec.start()
	cv.waitKey(10)
	roi_frames=[]
	beginning=0
	end=50

	while(True):

		image = frame.copy()
		roi,i = roi_extract(image)
		# print("roi"+str(time.time()))
		roi_frames.append(roi)

		video_magnification = threading.Thread(target=magnify_video, args=(beginning,end,))
		
		# print(len(roi_frames))
		if(len(roi_frames)==end):
			print(time.time())
			video_magnification.start()
			beginning=end
			end+=50


	



def roi_extract(image):

    global im
    #image = adjust_gamma(image,2.2)
    face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    #eye_cascade = cv.CascadeClassifier('C:\\opencv\\build\\share\\OpenCV\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')

    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    h_r,w_r,ch = image.shape
    roi= numpy.zeros((h_r,w_r,3), numpy.uint8)
    roi[:,:] = (255,255,255)
    i=0

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x,y,w,h) in faces:
        w1 = int(w*0.75)
        h1 = int(h*0.2)
        frame = cv.rectangle(image,(x+60,y),(x+w1,y+h1),(255,0,0),2)
        roi = image[y:y+h1, x+60:x+w1]

    #cv.imshow("Face Track",image)
    im=image
    cv.waitKey(10)

    return roi,i

def show_frame():
    global im
    frame = cv.flip(im, 1)


    cv2image   = cv.cvtColor(im, cv.COLOR_BGR2RGB)

    img   = Image.fromarray(cv2image).resize((480, 320))
    imgtk = ImageTk.PhotoImage(image = img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

def close_thread():
	# run.raise_exception()
	# run.setDaemon(True)
	# # rec.raise_exception()
	# rec.setDaemon(True)

	# exit()
	sys.exit()
	mainWindow.destroy()
	


if __name__ == '__main__':
	run = threading.Thread(target=start_func)
	startTime = time.time()
	print("start time" +  " "  + str(startTime))
	run.start()

	white 	= "#ffffff"
	lightBlue2 	= "#adc5ed"
	font = "Constantia"
	fontButtons = (font, 12)
	maxWidth  	= 600
	maxHeight 	= 420
    
    #Graphics window
	mainWindow = tk.Tk()
	mainWindow.title("Heart Rate Measurer | ver: 1.2")
	mainWindow.configure()
	mainWindow.geometry('%dx%d+%d+%d' % (maxWidth,maxHeight,0,0))
	mainWindow.resizable(0,0)
	mainWindow.protocol('WM_DELETE_WINDOW',close_thread)
	# mainWindow.overrideredirect(1)

	mainFrame = Frame(mainWindow)
	mainFrame.place(x=50, y=20)

	lmain = tk.Label(mainFrame)
	lmain.grid(row=0, column=0)

	show_frame()  #Display
	mainWindow.mainloop()  #Starts GUI

	cv.destroyAllWindows()


	