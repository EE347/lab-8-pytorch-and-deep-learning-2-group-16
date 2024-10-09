import cv2 
import os


input_folder = "./before_resize/neil"

output_folder = "./data/neil"
new_size = (64,64)

for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder,filename)
    #print(filepath)
    image = cv2.imread(filepath)
    
    resized = cv2.resize(image,new_size)
    output_path = os.path.join(output_folder,filename)
    print(output_path) #
    cv2.imwrite(filename,resized)