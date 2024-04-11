import cv2
import numpy as np
from pupil_apriltags import Detector

def get_distance(x1, y1, x2, y2):
     return np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
def width_in_frame(corners):
     w1 = get_distance(corners[0][0], corners[0][1], corners[3][0], corners[3][1])
     w2 = get_distance(corners[1][0], corners[1][1], corners[2][0], corners[2][1])
     return np.average([w1, w2])
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
at_detector = Detector(
   families="tag52h13",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

tag_width = 0.006 # in m
while(True): 
    ret, frame = vid.read() 
    if ret == True:
     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
     gray_uint8 = gray.astype(np.uint8)  # Convert to uint8
     
     cv2.imshow('frame', gray_uint8) 
     # print(gray.shape)
     params = np.load("./params.npy")

     fx = params[0,0]
     fy = params[1,1]
     cx = params[0,2]
     cy = params[1,2]

     detections = at_detector.detect(img=gray_uint8, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=tag_width)
     
     for detect in detections:
               print("tag_id: %s, center: %s" % (detect.tag_id, detect.center))
               width = width_in_frame(detect.corners)
               distance = tag_width * fx / width
               print("pose: %s" % (detect.pose_t))
               print("corners: %s" % (detect.corners))
               print("distance from formula: %s" % (distance))
               # image = plotPoint(image, detect.center, CENTER_COLOR)
               # image = plotText(image, detect.center, CENTER_COLOR, detect.tag_id)
               # for corner in detect.corners:
               #     image = plotPoint(image, corner, CORNER_COLOR)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
 
vid.release() 
cv2.destroyAllWindows() 

