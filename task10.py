import time
import cv2
import cv2
from picamera2 import Picamera2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

model = None

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = mobilenet_v3_small(weights=None, num_classes=2).to(device)

model.load_state_dict(torch.load("lab8/best_model.pth"))
model.eval()

cam = Picamera2()

cam.start()

cam.create_preview_configuration()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
width = 640
height = 480
vi = 0 # video index
pi = 0 # photo index

writer = cv2.VideoWriter()

def crop_face_from_image(frame_bgr):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.25, minNeighbors=5, minSize= (30,30), flags=cv2.CASCADE_FIND_BIGGEST_OBJECT)
        except:
            faces = []
        cropped = None

        if len(faces)>0:
            x,y,w,h = faces[0]
            cropped =  frame_bgr[y:y+h,x:x+w]
            cv2.rectangle(frame_bgr,(x,y),(x+w,y+h),(255,0,200))

            final_image = np.zeros(np.shape(frame_bgr),dtype=np.uint8)


            final_image[y:y+h,x:x+h] = cropped

            return True, final_image, [x,y,w,h]

        return False,[],[]

try:
        while True:
            frame = cam.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            key = cv2.waitKey(1) & 0xFF
            ret, cropped_frame, dims = crop_face_from_image(frame_bgr)

            if key == ord('q'):
                break


            if ret:
                x,y,w,h = dims
                cropped_frame = cropped_frame[y:y+h,x:x+h]
                model_input = cv2.resize(cropped_frame,(64,64))
                image_tensor = torch.tensor(model_input).reshape(-1, 3, 64, 64)
                output = model(image_tensor.to(torch.uint8))
                cv2.putText(frame_bgr,"Daniel" if output.item() == 0 else "Neil", (x,y), 0, 1,(255,0,0))

                pi += 1
            

            cv2.putText(frame_bgr,"q - quit, k - cropped image",(0,20),0,0.35,(255,255,255))
            cv2.imshow("Frame",frame_bgr)


finally:
        writer.release()
        cv2.destroyAllWindows()
        cam.stop()

cv2.destroyAllWindows()
cam.stop()