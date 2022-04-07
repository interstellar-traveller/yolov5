import torch
from torch._C import device
import torch.nn as nn
import torch
from PIL import Image
import cv2 as cv
from PIL import Image
import numpy

img = Image.open('F:\\0Desk\\yolo\\yolov5\\ball_sample.jpg')
# img.show()

img = numpy.array(img)

model = torch.hub.load('F:/0Desk/yolo/yolov5', 'custom', path='F:/0Desk/yolo/yolov5/best.pt', source='local')

def score_frame(frame, model):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model.to(device)
    results = model(frame) # returns a Detection Object, located in common.py
    labels = results.xyxyn[0][:, -1].numpy()
    cord = results.xyxyn[0][:, :-1].numpy()
    return labels, cord

def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.2: 
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
                    classes[int(labels[i])], \
                    (x1, y1), \
                    label_font, 0.9, rgb, 2) #Put a label over box.
    return frame

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = plot_boxes(score_frame(img, model), img)
cv.imshow("Prediction", img)


while True:
    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()