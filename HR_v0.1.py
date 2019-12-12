import sys
import os
sys.path.append("./modules")
import cv2 as cv
import numpy
from tkinter import *
from PIL import Image, ImageTk
import threading
import datetime
import time
from amplify_spatial_Gdown_temporal_ideal import amplify_spatial_Gdown_temporal_ideal
from sklearn.decomposition import FastICA
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg , NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import notify2
from matplotlib import style
style.use('ggplot')

"""
Global Variables:
"""
im = numpy.zeros((640,480,3),numpy.uint8) # for storing the current roi marked frame
frame = [] # array containing the recorded frames
close = 0 # to ensure all threads are closed once the MainWindow closes
roiColors = [] # array containing the roi of frames
MagnifyOver = 0 # for ensuring locking in videoMagnify function
MagnifyOver2 = 0 # for ensuring locking in videoMagnify function
lock = 0 # for ensuring locking of calculate function
UperCap = 5000 # for enusring memory management
amplifiedFrames = [] # Frames having amplified roi's to observe change in color signals 
faceCascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') # classifier for detecting face in an image
fps = 0 # frames per second
BPM = [] # array containing the bpm values averaged over the three color channels
HR = [] # array containing the final bpm values
count = [] # array containing the count of final bpm values, used for graph plotting
f= Figure(figsize=(6,3), dpi=100) # creating a figure, to be plotted on MainWindow
ax1 = f.add_subplot(1,1,1) # adding a subplot on the figure, to be used for graph plotting
lab = 0 # label on the MainWindow for current bpm value

notify2.init("Notifier") #initialising notification

"""
	Detect faces and roi in a frame.
	@params image- image in which the roi has to be detected.
	@return roi- The region of interest(the forehead of the subject) as a numpy array
			boolVal- True if roi was detected, false otherwise
	@contributors- Gandharv
"""
def faceDetect(image):
	global im, faceCascade

	#Convert the image to gray scale
	gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
	h_r,w_r,ch = image.shape

	#initialize roi as a numpy array
	roi = numpy.zeros((h_r,w_r,3), numpy.uint8)

	roi[:,:] = (255,255,255)
	
	#Detect the faces using haarcascade and store in faces.
	faces = faceCascade.detectMultiScale(gray, 1.3, 4)
	
	#Iterate over the detected faces and find the roi on the forehead of the subject
	for (x,y,w,h) in faces:
		w1 = int(w*0.30)
		w2 = int(w*0.70)
		h1 = int(h*0.10)
		h2 = int(h*0.22)
		#show the roi in the frame
		frame = cv.rectangle(image,(x+w1,y+h1),(x+w2,y+h2),(255,0,0),2)
		roi = image[y+h1:y+h2, x+w1:x+w2]

	im = image

	end = len(roi) - 1
	end2 = len(roi[end]) - 1

	#check if any roi was detected
	boolVal = not (roi[end][end2] == [255,255,255]).all()
	
	return roi,boolVal

"""
	Records the frames through webcam and stores them in the frame array.
	Updates the fps i.e. frames recorded per second.
	@contributors- Gaurav
"""

def record():
	global close, frame, fps

	# Capturing Video through webcam
	cap = cv.VideoCapture(0)

	initial = time.time()

	count = 0

	# For ensuring that the thread stops as soon as the MainWindow is closed
	while(close != 1):                            
		
		# Storing the current frame
		ret, nextFrame = cap.read()                

		# For ensuring whether the frame has been properly recorded or not
		if(ret == False or len(frame) > 1e4):      
			continue

		count += 1

		# Storing the current frame in frame array
		frame.append(nextFrame)	

		# Updating frames recorded per second
		fps = count//(time.time() - initial)

	cap.release()

	close = 1;
	print("finish record")
	exit()

"""
	Function used for magnifying the frames to observe the 
	changes in skin color for finding the heart signals from them.
	This function directly uses the modules created by
	some MIT students.
	@params colors: Array containg the ROI marked frames
	@contributors- Gaurav
"""
def videoMagnify(colors):
	global MagnifyOver, close, amplifiedFramesm, MagnifyOver2, fps

	# ensuring locking
	while(MagnifyOver == 1):
		continue

	MagnifyOver = 1

	fourcc = cv.VideoWriter_fourcc(*'MPEG')
	mag = cv.VideoWriter('video/output.avi',fourcc, 30, (120,96))

	# iterating over the frames
	for frame in colors:
		# ensuring that the MainWindow is still opened
		if(close == 1):
			break
		frame = cv.resize(frame, (120,96), interpolation = cv.INTER_AREA)
		mag.write(frame)
	
	mag.release()
	MagnifyOver = 0

	# ensuring locking
	while(MagnifyOver2 == 1):
		continue

	MagnifyOver2 = 1
	amplify_spatial_Gdown_temporal_ideal("video/output.avi","", 80,4,30.0/60.0,160.0/60.0,1.0,'rgb')

	vid = cv.VideoCapture("video/proc.mov")
	ret, Frame = vid.read()
	while(ret == True):
		amplifiedFrames.append(Frame)
		ret, Frame = vid.read()
	
	MagnifyOver2 = 0
	exit()

"""
	Plot the graph of heart rate in real-time.
	@params i- default parameter required by matplotlib.animation
	@contributors- Gandharv
"""
def gPage(i):
	global HR,count

	#plot the graph only if there is any value of Heart rate calculated
	if(len(HR)>0):
		ax1.clear()

		# plot values of Heart rate vs the numbers of heart rate calculated.
		ax1.plot(count,HR,lineWidth=1.2,linestyle = '-',marker='.',color = 'red',antialiased = False)
		ax1.set_ylabel('Heart Rate')
		ax1.set_ylim([35,240])
		if(count[-1]-count[0]<20):
			ax1.set_xlim([0,20])
		else:
			ax1.set_xlim([count[-21],count[-1]])


"""
	Show notification for high heart rate
	@params value- Measured heart rate value
"""
def show_notif(value):
	#Set the message to show in the notification.
	mess = "Your HR is "+ str(value) + ".\nPlease try to relax."
	n = notify2.Notification("High Heart Rate", message = mess)
	n.set_urgency(notify2.URGENCY_CRITICAL)
	n.show()



"""
	Function for doing all the processing and calculation on observed color signals
	to calcuate the final heart rate.
	@params amplified- Array containing the amplified frames
	@contributors- Aarish and Gaurav
"""
def calculate(amplified):
	global fps,BPM,HR,count
	
	# arrays for storing the mean values of respective color channel
	green = []
	red = []
	blue = []

	# iterating through the window of amplified frames
	for frame in amplified:

		# extracting all the three color channels of current frame
		b,g,r = cv.split(frame)

		# storing the mean value of current frame of each color channel in it's respective array
		meanG = numpy.mean(g[:,:])
		meanR = numpy.mean(r[:,:])
		meanB = numpy.mean(b[:,:])
		green.append(meanG)
		red.append(meanR)
		blue.append(meanB)

	# applying ICA technique to find the approximate source signals from observed signals
	ica = FastICA(n_components = 3, max_iter = 700, tol = 0.1)
	# matrix of observed signals
	X = [red, green, blue]
	# ICA giving a transform matrix of order of f(number of frames) * 3 
	W = ica.fit_transform(X)
	
	# declaring arrays of respective color channels to store the source signals
	SourceRed = []
	SourceGreen = []
	SourceBlue = []

	# multiplication of matrix W(given by ICA) with the observed signal matrix X
	for i in range(len(X[0])):
		SourceRed.append(W[0][0]*X[0][i] + W[0][1]*X[1][i] + W[0][2]*X[2][i])
		SourceGreen.append(W[1][0]*X[0][i] + W[1][1]*X[1][i] + W[1][2]*X[2][i])
		SourceBlue.append(W[2][0]*X[0][i] + W[2][1]*X[1][i] + W[2][2]*X[2][i])

	# frames per second currently
	fpsLocal = fps

	"""
	Algorithm followed ahead:
	Step 1: Applying fourier tranformation on the approximate source signals to convert them
	into frequency domain. 
	Step 2: Then storing the magnitude with respect to all frequencies
	in the magnitude array. 
	Step 3: Then storing the beats per minute values in the freq array.
	Beats per minute values are frequency values multiplied by 60.
	Frequency values are index multiplied by delta f, where delta f is
	1/(delta t * number of frames), where delta t is 1/(frames per second)
	Step 4: Then finding the beat with highest magnitude and storing that
	beat value as the bpm from respective source signal.
	"""

	# applying the above algorithm on Green channel
	FFT = numpy.fft.rfft(SourceGreen)
	magnitude = numpy.abs(FFT)
	freq = (fps/len(magnitude)*60)*numpy.arange(len(magnitude))
	index = numpy.where((freq >= 45) & (freq <= 240))
	Values = magnitude[index]
	maxVal = numpy.argmax(Values)
	GreenBPM = freq[index[0][0] + maxVal]

	# applying the above algorithm on Red channel
	FFT = numpy.fft.rfft(SourceRed)
	magnitude = numpy.abs(FFT)
	freq = (fps/len(magnitude)*60)*numpy.arange(len(magnitude))
	index = numpy.where((freq >= 45) & (freq <= 240))
	Values = magnitude[index]
	maxVal = numpy.argmax(Values)
	RedBPM = freq[index[0][0] + maxVal]

	# applying the above algorithm on Blue channel
	FFT = numpy.fft.rfft(SourceBlue)
	magnitude = numpy.abs(FFT)
	freq = (fps/len(magnitude)*60)*numpy.arange(len(magnitude))
	index = numpy.where((freq >= 45) & (freq <= 240))
	Values = magnitude[index]
	maxVal = numpy.argmax(Values)
	BlueBPM = freq[index[0][0] + maxVal]

	# averaging the three bpm's 
	avg = (RedBPM + BlueBPM + GreenBPM)/3
	BPM.append(avg)

	if(len(BPM) >= 10):
		# averaging the first 10 unused bpm's to find the final bpm
		avgval = sum(BPM[:10])/10	
		avgval = round(avgval,2)
		# storing the final bpm value in HR array
		HR.append(avgval)
		# storing the count of values for graph plotting
		if len(count)>0:
			count.append(count[-1]+1)
		else:
			count.append(0)
		
		try:
			# updating the label on the MainWindow with the current bpm value
			lab.config(text = "Heart Rate = "+str(avgval)+"bpm")
			# storing the final bpm in a file
			log_file(avgval)
			# printing the final bpm
			print("BPM = ",avgval)

			# deleting the recently used 10 values of bpm array for memory management
			del BPM[:10]
			# deleting the first 10 final bpm values from HR array and respective counts
			# from count array for memory management
			if(len(HR)>20):
				HR = HR[-21:]
				count = count[-21:]

			if avgval > 150:
				show_notif(avgval)

		except:
			exit()


"""
	Initiating the thread for capturing the video through webcam
	Detecting the face of the current frame
	Initiating the thread for magnifying the ROI's for observing changes in skin color
	Initiating the thread for calculating the heart rate
	@contributors- Gaurav
"""
def execute():
	global close, roiColors, UperCap, amplifiedFrames, frame, lock

	rec = threading.Thread(target=record)
	rec.start()
	to = time.time()
	lastUsed = 0
	lastUsed2 = 0

	# For ensuring that the thread stops as soon as the MainWindow is closed
	while(close != 1):
		
		# If no frames has been recorded, do nothing
		if(len(frame) == 0):
			continue

		# popping the current frame from frame array
		img = frame.pop()
		# detecting face in the current frame and further finding ROI
		roi, boolVal = faceDetect(img)
		
		# If face is not detected, do nothing
		if(boolVal == False):
			continue
		
		#Preserve frames in execute if upercap is achieved
		if(len(roiColors) != UperCap):	
			roiColors.append(roi)

		# if there are >= 50 unused frames
		if(len(roiColors) > 0 and (len(roiColors) - lastUsed) >= 50):
			# storing first 50 unused frames in another array
			colors = roiColors[lastUsed:lastUsed + 50]
			# thread for magnifying frames of color array
			evm = threading.Thread(target=videoMagnify, args=(colors,))
			# initiating the thread
			evm.start()
			# marking the current frames as used
			lastUsed += 50

			# releasing 1000 frames from roiColors array to ensure memory management
			if(len(roiColors) >= 1000):
				del roiColors[:1000]
				lastUsed = 0

		# ensuring if there are more than 300 amplified frames in amplifiedFrames array
		# print(len(amplifiedFrames))
		if(len(amplifiedFrames) > 300):
			# for ensuring locking of calculate thread
			while(lock == 1):
				continue

			lock = 1
			# taking a window of 300 frames
			amplified = amplifiedFrames[0:300]
			# formation of thread for calculating heart rate
			monitor = threading.Thread(target=calculate, args=(amplified,))
			# initiating the thread
			monitor.start()
			# moving the window ahead by 5 frames
			del amplifiedFrames[:5]

			lock = 0


	print("finish execute")
	exit()

"""
	Display the frames captured by the camera
	@contributors- Gandharv
"""
def showFrame():
	global im

	cv2image   = cv.cvtColor(im, cv.COLOR_BGR2RGB)
	img = Image.fromarray(cv2image).resize((480, 320))
	imgtk = ImageTk.PhotoImage(image = img)
	lmain.imgtk = imgtk
	lmain.configure(image=imgtk)
	lmain.after(10, showFrame)


"""
	Initialize all components of GUI for the application
	@return mainFrame- Frame for dispalying the frames caputed by the camera
			mainWindow- tkinter window for the application
	@contributors- Gandharv and Kanupriya
"""
def setWindow():
	global lab
	
	# Set up the Graphics window 
	mainWindow = Tk()
	mainWindow.title("Heart Rate Measurer | ver: 0.1")
	mainWindow.configure()
	mainWindow.geometry('%dx%d+%d+%d' % (720,1050,0,0))
	mainWindow.resizable(0,0)

	#Set the frame to show the video captured
	mainFrame = Frame(mainWindow)
	mainFrame.place(x=100, y=20)

	#Label to show the heart rate of the subject
	lab = Label(mainWindow,text = "Heart Rate = --bpm",font=("Helvetica", 15))
	lab.place(x = 220,y = 360)

	# Initialize the subplot for graph
	ax1.clear()
	ax1.set_ylim([35,240])
	ax1.set_xlim([0,20])
	ax1.set_ylabel('Heart Rate')
	ax1.grid(True)

	# Set up the canvas for showing the graph of heart rate
	canvas = FigureCanvasTkAgg(f,mainWindow)
	canvas.draw()
	canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand = True)
	canvas._tkcanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand = True)
	canvas._tkcanvas.place(x = 60, y= 400)
	

	return mainFrame,mainWindow

"""
	Function for writing the final bpm values in an excel sheet
	@params value- The final bpm value
	@contributors- Gaurav and Kanupriya
"""
def log_file(value):
	# ensuring that the MainWindow is still opened
	if(close == 1):
		return
	# writing the current time and current final bpm in the excel sheet
	now = datetime.datetime.now()
	row = str(now) + ", " + str(value) + "\n"
	fd.write(row)


if __name__ == "__main__":

	#Open the csv file to log the Heart rate values
	fd = open("HR_values.csv", "a")

	if(not os.path.isdir("./video")):
		os.mkdir("video")

	
	# Make a thread for parallely running the execute function
	run = threading.Thread(target=execute)
	run.start()

	mainFrame,mainWindow = setWindow()

	# Capture video frames
	lmain = Label(mainFrame)
	lmain.grid(row=0, column=0)


	showFrame()

	# Plot the graph in real-time
	ani = animation.FuncAnimation(f, gPage, interval=1000)

	#Render the GUI
	mainWindow.mainloop()

	# Stop all the threads when the window is closed
	close = 1;
	cv.destroyAllWindows()

	# Wait for all threads to stop
	time.sleep(2)
	fd.close()