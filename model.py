import torch
import numpy
import darknet
import cv2


# Training section.


img = cv2.imread("dumb.jpg")
img_tensor = torch.from_numpy(img)
img_tensor = img_tensor.float()
img_tensor_diff_perc = img_tensor.permute(
    2, 0, 1
)  #  ***********important*********** function to change the order of elements in the memory rather than only shape
img_tensor_diff_perc_unsqueezed = torch.unsqueeze(img_tensor_diff_perc, 0)
img_model = darknet.Darknet("yolov4.cfg", img_tensor_diff_perc_unsqueezed)

predicted_boxes = img_model.get_boxes()
print(0)
