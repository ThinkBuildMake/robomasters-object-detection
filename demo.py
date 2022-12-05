import torch
import cv2
from kalmanfilter import KalmanFilter

# function for getting center of armor plate, wont work if there are multiple objects detected
def getXY(xy):
    x = (xy['xmin'][0] + xy['xmax'][0]) / 2
    y = (xy['ymin'][0] + xy['ymax'][0]) / 2
    return (x,y)

# Models and Filters
model = torch.hub.load('yolov5/', 'custom', 
    path='yolov5\\runs\\train\\motion_sensitive_model\\weights\\best.pt', 
    source='local')
kf = KalmanFilter()

# Setup camera
cam_feed = cv2.VideoCapture(0)
cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)


while True:    
    # yolo detection
    ret, img = cam_feed.read()   
    pred = model(img, size=640)

    output_img = pred.render()[0]

    # Kalman filter prediction
    if(len(pred.pandas().xyxy[0]) > 0):
        x, y = getXY(pred.pandas().xyxy[0])

        xpred, ypred = kf.predict(x, y)

        # draw the circle
        cv2.circle(output_img, [int(xpred), int(ypred)], 10, color=(0, 165, 255), thickness=3)
        
    # display the frame
    cv2.imshow("", output_img)     
    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

cam_feed.release()
cv2.destroyAllWindows()
