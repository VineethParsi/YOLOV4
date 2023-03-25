import torch
import math
from collections import OrderedDict


class EmptyModule(torch.nn.Module):
    def __init__(self) -> None:
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


class DetectionLayer(torch.nn.Module):
    def __init__(self, anchors, x) -> None:
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.x = input


class Darknet(torch.nn.Module):
    def __init__(self, filename) -> None:
        super().__init__(Darknet, self)
        self.filename = filename

    def get_boxes(self):
        blocks = self.parse_cfg(self.filename)
        module_list = self.modules(blocks)
        detections = self.forwardpass(modules_list=module_list, blocks=blocks, x=self.x)
        return detections

    def modules(blocks):
        modules = []
        net_info = blocks[0]
        output_filters = []
        prev_filters = 3
        for i, block in enumerate(blocks[1:]):

            if block[type] == "convolution":
                output_filter = block["filters"]
                conv2d = torch.nn.Conv2d(
                    input_channels=prev_filters,
                    out_channels=output_filter,
                    kernel_size=block["size"],
                    stride=block["stride"],
                    padding=block["pad"],
                )

                ############################ write activation type code here #########################

                activation = torch.nn.Mish()
                prev_filters = output_filter
                batchNorm = torch.nn.BatchNorm2d(num_features=prev_filters)
                output_filter = prev_filters
                output_filters[i] = output_filter
                OrderedDic = OrderedDict()
                OrderedDic["conv2d"] = conv2d
                OrderedDic["activation"] = activation
                OrderedDic["batchNorm"] = batchNorm
                modules.append(torch.nn.sequential(OrderedDic))
                prev_filters = output_filter

            elif block[type] == "route":  #
                layers = block["layers"].split(",")
                layers = [int(x.strip()) for x in layers]
                layers = [x if x < 0 else (x - i) for x in layers]
                modules.append(EmptyModule())
                prev_filters = 0
                for d in layers:
                    prev_filters += output_filters[i + d]

                output_filter = prev_filters
                output_filters[i] = output_filter

            elif block[type] == "shortcut":
                output_filters[i] = output_filter
                modules.append(EmptyModule())

            elif block[type] == "maxpool":
                maxpool2d = torch.nn.MaxPool2d(
                    stride=block["stride"], kernel_size=block["size"]
                )
                modules.append(maxpool2d)
                output_filters[i] = output_filter

            elif block[type] == "upsample":
                stride = block["stride"]
                upsamplelayer = torch.nn.Upsample(scale_factor=stride, mode="nearest")
                modules.append(upsamplelayer)
                output_filters[i] = output_filter

            elif block[type] == "yolo":
                masks = block["mask"].split(",")
                masks = [int(i) for i in masks]
                anchors = block["anchors"].strip().split(",")
                anchors = [
                    (int(anchors[i]), int(anchors[i + 1]))
                    for i in range(0, anchors.len, 2)
                ]
                anchors = [anchors[i] for i in masks]
                detectionlayer = DetectionLayer(anchors=anchors)
                modules.append(detectionlayer)
                output_filters[i] = output_filter
        return modules, net_info

    def parse_cfg(cfg):
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
                block[key] = val
        blocks.append(block)
        return blocks

    def x_trans(x):
        return torch.nn.Sigmoid(x)

    def y_trans(x):
        return torch.nn.Sigmoid(x)

    def h_trans(x):
        return math.exp(x)

    def w_trans(x):
        return math.exp(x)

    def c_trans(x):
        return (torch.argmax(x)).item()  # write argmax function

    def detections_func(self, features, anchors, classes, input_len, input_wid):
        n_rows = features.size()[0]
        n_columns = features.size()[1]
        cell_width_pixels = input_len / n_columns
        cell_height_pixels = input_wid / n_rows

        boxes_array = torch.clone(features)
        boxes_array = boxes_array.view(-1, 85)
        boxes_array[:, 0] = boxes_array[:, 0].apply_(self.x_trans)
        boxes_array[:, 1] = boxes_array[:, 1].apply_(self.y_trans)
        boxes_array[:, 2] = boxes_array[:, 2].apply_(self.h_trans)
        boxes_array[:, 3] = boxes_array[:, 3].apply_(self.w_trans)
        boxes_array[:, 5] = boxes_array[:, 5:].apply_(self.c_trans)

        for i in range(boxes_array.size()[0]):

            anchor_no = (i + 1) % 3
            anchor_no += 1

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

        for i, module in enumerate(modules_list):

            block = blocks[i]

            if block["type"] == "convolution":
                x = module(x)
                outputs.append(x)

            elif block["type"] == "shortcut":

                negativelayerdist = int(blocks[i]["from"].strip())
                # activation = int(blocks[i]["activation"])
                x = x + outputs[i + negativelayerdist]
                ################################################## write code for activation function##############################################
                x = torch.nn.Mish(x)
                outputs.append(x)

            elif block["type"] == "route":
                layers = block["layers"].strip().split(",")
                layers = [int(i) for i in layers]
                if layers[0] < 0:
                    layers[0] += i
                layer = outputs[layers[0]]
                for layer in range(0, len(layers)):
                    if layer < 0:
                        layer = i + layer

                    x = torch.concat((x, outputs[layer]), dim=2)

                outputs[i] = x

            elif block["type"] == "maxpool":
                x = module(x)
                outputs[i] = x

            elif block["type"] == "upsample":
                x = module(x)
                outputs[i] = x

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
