import cv2
import numpy as np
from pupil_apriltags import Detector


# import the opencv library 
import cv2 
  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
at_detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

tag_width = 0.013 # in m
while(True): 
      
    ret, frame = vid.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
    gray_uint8 = gray.astype(np.uint8)  # Convert to uint8
    
    cv2.imshow('frame', gray_uint8) 
    # print(gray.shape)
    fx, fy, cx, cy = (1587.5427555739097, 1591.9066393069938, 865.5419508049021, 680.0115264114966)
    detections = at_detector.detect(img=gray_uint8, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=tag_width)
    
    for detect in detections:
            print("tag_id: %s, center: %s" % (detect.tag_id, detect.center))
            width = detect.corners[0] - detect.corners[1]
            distance = tag_width * fx / width
            print("distance from pose: %s" % (detect.pose_t[2][0]))
            # print("distance from formula: %s" % (distance))
            # image = plotPoint(image, detect.center, CENTER_COLOR)
            # image = plotText(image, detect.center, CENTER_COLOR, detect.tag_id)
            # for corner in detect.corners:
            #     image = plotPoint(image, corner, CORNER_COLOR)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
 
vid.release() 
cv2.destroyAllWindows() 

