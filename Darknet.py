import torch
import math
from collections import OrderedDict


class EmptyModule(torch.nn.Module):
    def __init__(self) -> None:
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


class DetectionLayer(torch.nn.Module):
    def __init__(self, anchors) -> None:
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(torch.nn.Module):
    def __init__(self, filename, input_img) -> None:
        super(Darknet, self).__init__()
        self.filename = filename
        self.x = input_img

    def get_boxes(self):
        blocks = self.parse_cfg(self.filename)
        module_list, net_info = self.modules(blocks)
        detections = self.forwardpass(modules_list=module_list, blocks=blocks, x=self.x)
        return detections

    def modules(self, blocks):
        modules = []
        net_info = blocks[0]
        output_filters = []
        prev_filters = 3
        for i, block in enumerate(blocks[1:]):
            if block["type"] == "convolutional":
                output_filter = int(block["filters"])
                pad = (int(block["size"]) - 1) // 2
                conv2d = torch.nn.Conv2d(
                    in_channels=prev_filters,
                    out_channels=int(output_filter),
                    kernel_size=int(block["size"]),
                    stride=int(block["stride"]),
                    padding=pad,
                )

                ############################ write activation type code here #########################

                activation = torch.nn.Mish()
                prev_filters = output_filter
                batchNorm = torch.nn.BatchNorm2d(num_features=prev_filters)
                output_filter = prev_filters
                output_filters.append(output_filter)
                OrderedDic = OrderedDict()
                OrderedDic["conv2d"] = conv2d
                OrderedDic["activation"] = activation
                OrderedDic["batchNorm"] = batchNorm
                modules.append(torch.nn.Sequential(OrderedDic))
                prev_filters = output_filter

            elif block["type"] == "route":  #
                layers = block["layers"].split(",")
                layers = [int(x.strip()) for x in layers]
                layers = [x if x < 0 else (x - 1 - i) for x in layers]
                modules.append(EmptyModule())
                prev_filters = 0
                for d in layers:
                    prev_filters += output_filters[i + d]

                output_filter = prev_filters
                output_filters.append(output_filter)

            elif block["type"] == "shortcut":
                output_filters.append(output_filter)
                modules.append(EmptyModule())

            elif block["type"] == "maxpool":
                pad = (int(block["size"]) - 1) // 2
                maxpool2d = torch.nn.MaxPool2d(
                    stride=int(block["stride"]),
                    kernel_size=int(block["size"]),
                    padding=pad,
                )
                modules.append(maxpool2d)
                output_filters.append(output_filter)

            elif block["type"] == "upsample":
                stride = block["stride"]
                upsamplelayer = torch.nn.Upsample(scale_factor=stride, mode="nearest")
                modules.append(upsamplelayer)
                output_filters.append(output_filter)

            elif block["type"] == "yolo":
                masks = block["mask"].split(",")
                masks = [int(i) for i in masks]
                anchors = block["anchors"].strip().split(",")
                anchors = [
                    (int(anchors[i]), int(anchors[i + 1]))
                    for i in range(0, len(anchors), 2)
                ]
                anchors = [anchors[i] for i in masks]
                detectionlayer = DetectionLayer(anchors=anchors)
                modules.append(detectionlayer)
                output_filters.append(output_filter)
        return modules, net_info

    def parse_cfg(self, cfg):
        file_input_stream = open(cfg, "r")
        lines = file_input_stream.read().split("\n")

        lines = [x for x in lines if len(x) > 0]  # get rid of empty lines
        lines = [x for x in lines if x[0] != "#"]  # get rid of comments

        block = {}  # modules description blocks
        blocks = []  # list of blocks

        for line in lines:
            if line[0] == "[":
                if bool(block):
                    blocks.append(block)
                block = {}
                block["type"] = line[1:-1]
            else:
                key, val = line.split("=")
                block[key.strip()] = val.strip()
        blocks.append(block)
        return blocks

    def x_trans(self, x):

        return 1 / (1 + math.exp(-1 * x))

    def y_trans(self, x):
        return 1 / (1 + math.exp(-1 * x))

    def h_trans(self, x):
        return math.exp(x)

    def w_trans(self, x):
        return math.exp(x)

    def c_trans(self, x):
        return torch.argmax(x)  # write argmax function

    def detections_func(self, features, anchors, classes, input_len, input_wid):
        n_rows = features.size()[0]
        n_columns = features.size()[1]
        cell_width_pixels = input_len / n_columns
        cell_height_pixels = input_wid / n_rows

        boxes_array = torch.clone(features)
        boxes_array = boxes_array.view(-1, 85)
        boxes_array[:, 0] = boxes_array[:, 0].detach().apply_(self.x_trans)
        boxes_array[:, 1] = boxes_array[:, 1].detach().apply_(self.y_trans)
        boxes_array[:, 2] = boxes_array[:, 2].detach().apply_(self.h_trans)
        boxes_array[:, 3] = boxes_array[:, 3].detach().apply_(self.w_trans)
        for i in range(boxes_array.size()[0]):
            boxes_array[i, 5] = self.c_trans(boxes_array[i, 5:])

        for i in range(boxes_array.size()[0]):
            anchor_no = (i + 1) % 3
            # anchor_no += 1    #need to check the code for anchors here#############################################
            row = (((i + 1) / 3) - 1) / n_columns
            col = (((i + 1) / 3) - 1) % n_columns
            boxes_array[i][0] += (col) * cell_width_pixels
            boxes_array[i][1] += (row) * cell_height_pixels
            boxes_array[i][2] *= anchors[anchor_no][0]
            boxes_array[i][3] *= anchors[anchor_no][1]

        return boxes_array

    def forwardpass(self, modules_list, blocks, x):
        outputs = []
        detections = []
        yolo = 0
        for i, module in enumerate(modules_list):

            block = blocks[i + 1]
            print(block["type"], "{} block started".format(i))

            if block["type"] == "convolutional":
                x = module(x)
                outputs.append(x)

            elif block["type"] == "shortcut":

                negativelayerdist = int(block["from"].strip())
                # activation = int(blocks[i]["activation"])
                x = x + outputs[i + negativelayerdist]
                ################################################## write code for activation function##############################################
                mish = torch.nn.Mish()
                x = mish(x)
                outputs.append(x)

            elif block["type"] == "route":
                layers = block["layers"].strip().split(",")
                layers = [int(i) for i in layers]
                print(layers[0])
                if layers[0] < 0:
                    layers[0] += i
                print(layers[0])
                x = outputs[layers[0]]
                for layer in range(1, len(layers)):
                    if layers[layer] < 0:
                        layer = i + layers[layer]

                    x = torch.concat((x, outputs[layer]), dim=1)

                outputs.append(x)

            elif block["type"] == "maxpool":
                x = module(x)
                outputs.append(x)

            elif block["type"] == "upsample":
                x = module(x)
                outputs.append(x)

            elif block["type"] == "yolo":
                anchors = module.anchors
                classes = block["classes"]
                image_input_len = (
                    608  #######################give image input length here
                )
                image_input_wid = (
                    608  #######################give image input width here
                )
                detections.append(
                    self.detections_func(
                        x,
                        anchors,
                        classes=classes,
                        input_len=image_input_len,
                        input_wid=image_input_wid,
                    )[:, 0:6]
                )
                outputs.append(0)
                print("above shape is considered for detections", yolo)  # tttt
                yolo += 1  # tttt
                continue  # tttt

            print(x.size())  # tttt
        return detections

    def train(self, x, labels):
        x.requires_grad_(True)
        y = self.forwardpass(x)
        error = labels - y

        torch.optim.Adam(error)
