import cv2
from picamera2 import Picamera2
import numpy as np

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


            elif key == ord('k'):
                
                
                if ret:
                    print(f"saving image {pi}")
                    x,y,w,h = dims
                    cropped_frame = cropped_frame[y:y+h,x:x+h]
                    cv2.imwrite(f'./lab-6-python-and-opencv-2-group-16/neil/photo-{pi}.png', cropped_frame)

                    pi += 1
            

            cv2.putText(frame_bgr,"q - quit, k - cropped image",(0,20),0,0.35,(255,255,255))
            cv2.imshow("Frame",frame_bgr)


finally:
        writer.release()
        cv2.destroyAllWindows()
        cam.stop()

cv2.destroyAllWindows()
cam.stop()