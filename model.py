import torch
import numpy


def openCfgFile(cfg):
    fileReader = open(cfg, "r")
    
    current_line = fileReader.readline()
    while current_line != "":

        current_line = current_line[1,-2]

        if current_line == "convolution":
            
            batch_normalize = int(fileReader.readline()[0:-1])
            filters = int(fileReader.readline()[0:-1])
            size = int(fileReader.readline()[0:-1])
            stride = int(fileReader.readline()[0:-1])
            pad = int(fileReader.readline()[0:-1])
            activation = fileReader.readline()[0:-1]
            current_line = fileReader.readline()

           
        elif last_line =="route":
            
            layers = int(fileReader.readline()[0:-1])
            current_line = fileReader.readline()

        elif last_line =="net":
            current_line  = fileReader.readline()
            while current_line[0] != "[" and current_line != "":
                process(current_line)
                current_line = fileReader.readline()

        elif last_line =="shortcut":
            from_position = int(fileReader.readline()[0:-1])
            activation = fileReader.readline()[0:-1]
            current_line = fileReader.readline()


        elif last_line =="maxpool":
            stride = int(fileReader.readline()[0:-1])
            size = int(fileReader.readline()[0:-1])
            current_line = fileReader.readline()

        elif last_line =="upsample":
            stride = int(fileReader.readline()[0:-1])
            current_line = fileReader.readline()

        elif last_line =="yolo":
            mask = 3,4,5
            anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
            classes=int(fileReader.readline()[0:-1])
            num=int(fileReader.readline()[0:-1])
            jitter=float(fileReader.readline()[0:-1])
            ignore_thresh = float(fileReader.readline()[0:-1])
            truth_thresh = int(fileReader.readline()[0:-1])
            scale_x_y = float(fileReader.readline()[0:-1])
            iou_thresh=float(fileReader.readline()[0:-1])
            cls_normalizer=float(fileReader.readline()[0:-1])
            iou_normalizer=float(fileReader.readline()[0:-1])
            iou_loss=fileReader.readline()[0:-1]
            nms_kind=fileReader.readline()[0:-1]
            beta_nms=float(fileReader.readline()[0:-1])
            current_line = fileReader.readline()
        