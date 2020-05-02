import cv2
import numpy as np

windowName='Object Tracker'
class ObjectTracker(object):
	def __init__(self, scaling_factor=1.0):
		self.cap=cv2.VideoCapture(0)

		ret,self.frame = self.cap.read()

		self.scaling_factor = scaling_factor

		self.frame = cv2.resize(self.frame, None, fx=scaling_factor, fy=self.scaling_factor, interpolation=cv2.INTER_AREA)
#		img = np.zeros((512,512,3), np.uint8)
		cv2.namedWindow(windowName)
#		print(self.frame)
#		cv2.imshow('Object Tracker', self.frame)
		cv2.setMouseCallback(windowName, self.mouse_event)
#		cv2.imshow('Object Tracker', self.frame)
		self.selection=None
		self.drag_start=None
		self.tracking_state=0

	def mouse_event(self, event, x, y, flags, param):
#		print(self.frame)
		x, y = np.int16([x,y]) #convert 'x' and 'y' co-ordinates into 16 bit integers
#		print(x,y)
		if event == cv2.EVENT_LBUTTONDOWN: #check if a mouse button is pressed
			self.drag_start = (x,y)
			self.tracking_state = 0

		if self.drag_start:				   #check if the user started selecting the region
			if flags & cv2.EVENT_FLAG_LBUTTON:
				h,w = self.frame.shape[:2] #Extract the dimensions of the frame : 'h' = height & 'w' = width
#				print(h,w)
				xi,yi=self.drag_start      #Get the initial position of the selection
#				print(x,y,xi,yi)
				x0,y0=np.maximum(0,np.minimum([xi,yi],[x,y])) #Get the max and min values
				x1,y1=np.minimum([w,h],np.maximum([xi,yi],[x,y]))
#				print(x,y,xi,yi,x0,y0,x1,y1)
				self.selection=None        #Reset the selection variable

				if x1-x0 > 0 and y1-y0 > 0: #Finalise the selection
					self.selection = (x0, y0, x1, y1)
			else:						   #If selection is done, start tracking.
				self.drag_start = None
				if self.selection is not None:
					self.tracking_state = 1

#Method to start tracking the object
	def start_tracking(self): 			   #Iterate until user presses 'Esc' key
#		print(self.frame)
		while True:
			ret, self.frame = self.cap.read() #Capture frame from webcam
			self.frame = cv2.resize(self.frame, None, fx=self.scaling_factor, fy=self.scaling_factor, interpolation = cv2.INTER_AREA)
#			print(self.frame)
			vis = self.frame.copy()        #Create a copy of the frame
#			print(vis)
			hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV) #Convert the frame to HSV colorspace

			mask =cv2.inRange(hsv, np.array((0.,60.,32.)),np.array((180.,255.,255.))) #Create the mask based on predefined thresholds
#			cv2.imshow(windowName, mask)
#			vis=cv2.flip(vis, 1)
			if self.selection:
				x0,y0,x1,y1 = self.selection
#				print((x0,y0),(x1,y1))
				self.track_window=(x0,y0,x1-x0,y1-y0)
				print(self.track_window)
				hsv_roi=hsv[y0:y1,x0:x1]
				mask_roi=mask[y0:y1,x0:x1]
#				print(hsv_roi,mask_roi)
				hist = cv2.calcHist([hsv_roi],[0],mask_roi, [16],[0,180])
				cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
				self.hist = hist.reshape(-1)
				vis_roi = vis[y0:y1,x0:x1]
				cv2.bitwise_not(vis_roi, vis_roi)
				vis[mask==0] = 0
#				print(vis)
			if self.tracking_state ==1:
				self.selection=None
				hsv_backproj = cv2.calcBackProject([hsv],[0],self.hist,[0,180],1)
				hsv_backproj &= mask
				term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
				track_box, self.track_window =cv2.CamShift(hsv_backproj,self.track_window,term_crit)
#				print(track_box)
				cv2.ellipse(vis, track_box, (0, 0, 255), 5)
#				cv2.rectangle(vis,(x0,y0),(x1,y1),(0,0,255),5) 
#				cv2.imshow(windowName, img)
#				print(track_box, hsv_backproj)
			cv2.imshow(windowName, vis)		#Show the output Live video
#			cv2.imshow('Hello', hsv_roi)
			c = cv2.waitKey(5)      #Stop if the user hits the 'Esc' key
			if c == 27:
				break
		cv2.destroyAllWindows()
if __name__ == '__main__':
	ObjectTracker().start_tracking()
