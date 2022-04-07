import numpy
import torch
from torch._C import device
import torch.nn as nn
import torch
from PIL import Image
import cv2 as cv

# stream = cv.VideoCapture('F:/0Desk/yolo/yolov5/ball_demo.mp4')
stream = cv.VideoCapture("http://10.51.65.213:1181/stream.mjpg")

# model = nn.Module()
# print(torch.load('best.pt'))
# model.load_state_dict(torch.load('best.pt')['model'])

model = torch.hub.load('F:/0Desk/yolo/yolov5', 'custom', path='F:/0Desk/yolo/yolov5/best.pt', source='local')

def score_frame(frame, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    results = model(frame)
    labels = results.xyxyn[0][:, -1].detach().cpu().numpy()
    cord = results.xyxyn[0][:, :-1].detach().cpu().numpy()
    return labels, cord

def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    # frame = Image.fromarray(frame, 'RGB')
    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.5: 
            continue
        x1 = int(row[0]*x_shape)
        y1 = int(row[1]*y_shape)
        x2 = int(row[2]*x_shape)
        y2 = int(row[3]*y_shape)
        rgb = (0, 255, 0) # color of the box
        classes = model.names # Get the name of label index
        label_font = cv.FONT_HERSHEY_SIMPLEX #Font for the label.
        cv.rectangle(frame, \
                      (x1, y1), (x2, y2), \
                       rgb, 2) #Plot the boxes
        cv.putText(frame,\
                    classes[int(labels[i])] + str(row[4]), \
                    (x1, y1), \
                    label_font, 0.6, rgb, 2) #Put a label over box.
    return frame

cv.namedWindow("Tracking", cv.WINDOW_NORMAL)
# cv.setWindowProperty("Tracking", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cv.setWindowProperty("Tracking", int(cv.WND_PROP_FULLSCREEN/2), int(cv.WINDOW_FULLSCREEN/2))

# pause = True

while True:
    ret, frame = stream.read()
    if ret == False:
        print("Img Gay!")
        break
    cv.imshow("mjpg", frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = plot_boxes(score_frame(frame, model), frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.imshow("Tracking", frame)
    if cv.waitKey(1) == 27:
        break
    
    # while pause:
    #     if cv.waitKey(100) & 0xFF == ord('q'):
    #         pause = False
    
stream.release()
cv.destroyAllWindows()