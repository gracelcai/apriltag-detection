import numpy as np
import cv2 
 
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

width = 22 # in mm
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
vid = cv2.VideoCapture('calibration_vid.mp4') 
num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
print("frames:", num_frames, "type:", type(num_frames))

fps = vid.get(cv2.CAP_PROP_FPS)
duration = int(num_frames / fps)
second = 0
frame_index = 0
interval = int(num_frames / 50)
frame_count = 0
while(frame_index <= num_frames and frame_count < 50): 
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = vid.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
        gray_uint8 = gray.astype(np.uint8)  # Convert to uint8
        
        # Find the chess board corners
        success, corners = cv2.findChessboardCorners(gray, (11,8), None)
        
        # If found, add object points, image points (after refining them)
        if success == True:
            print("found corners")
            objpoints.append(objp)
        
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(frame, (11,8), corners2, ret)
            cv2.imshow('img', frame)
            frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    frame_index += 1

vid.release() 
cv2.destroyAllWindows() 
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

fx = K[0,0]
fy = K[1,1]
cx = K[0,2]
cy = K[1,2]

params = (fx, fy, cx, cy)

np.save("./params", K)

print()
print('all units below measured in pixels:')
print('  fx = {}'.format(K[0,0]))
print('  fy = {}'.format(K[1,1]))
print('  cx = {}'.format(K[0,2]))
print('  cy = {}'.format(K[1,2]))
print()
print('pastable into Python:')
print('  fx, fy, cx, cy = {}'.format(repr(params)))
print()

