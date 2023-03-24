import torch


class EmptyModule(torch.nn.Module):
    def __init__(self) -> None:
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


class DetectionLayer(torch.nn.Module):
    def __init__(self, anchors) -> None:
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


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
            batchNorm = torch.nn.BatchNorm2d(num_features=output_filter)
            output_filters[i] = output_filter
            OrderedDict = OrderedDict()
            OrderedDict["conv2d"] = conv2d
            OrderedDict["activation"] = activation
            OrderedDict["batchNorm"] = batchNorm
            modules.append(torch.nn.sequential(OrderedDict))
            prev_filters = output_filter

        elif block[type] == "route":  #
            layers = block["layers"].split(",")
            layers = [int(x) if x < 0 else i - int(x) for x in layers if x < 0]
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
            anchors = block["anchors"].split(",")
            anchors = [
                (int(anchors[i]), int(anchors[i + 1])) for i in range(0, anchors.len, 2)
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
    return torch.nn.Sigmoid(x) + cell_x


def y_trans(x):
    return torch.nn.Sigmoid(x) + cell_x


def h_trans(x):
    return anchor_box_height * (e ^ x)


def w_trans(x):
    return anchor_box_height * (e ^ x)


def c_trans(x):
    return argmax(x)  # write argmax function


def detections_func(features, anchors, classes, input_len, input_wid):

    boxes_array = torch.clone(features)
    boxes_array = boxes_array.view(-1, 85)
    boxes_array[:, 0] = boxes_array[:, 0].apply_(x_trans)
    boxes_array[:, 1] = boxes_array[:, 1].apply_(y_trans)
    boxes_array[:, 2] = boxes_array[:, 2].apply_(h_trans)
    boxes_array[:, 3] = boxes_array[:, 3].apply_(w_trans)
    boxes_array[:, 4] = boxes_array[:, 4:].apply_(c_trans)
    return boxes_array


def forwardpass(modules_list, blocks, x):
    outputs = []
    detections = []

    for i, module in enumerate(modules_list):

        block = blocks[i]

        if block["type"] == "convolution":
            x = module.forward(x)
            outputs.append(x)
        elif block["type"] == "shortcut":
            negativelayerdist = int(blocks[i]["from"])
            activation = int(blocks[i]["activation"])
            x = x + outputs[i + negativelayerdist]
            ################################################## write code for activation function##############################################
            x = torch.nn.Mish(x)
            outputs.append(x)

        elif block["type"] == "route":
            layers = block["layers"].strip().split(",")
            layers = [int(i) for i in layers]
            x = layers[0]
            for l in range(1, len(layers)):
                if l < 0:
                    layerindex = i + l
                else:
                    layerindex = l
                x = torch.concat((x, outputs[layerindex]), dim=2)

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
            image_input_len = 3                        #######################give image input length here
            image_input_wid = 3                        #######################give image input width here
            detections.append(detections_func(x,anchors, classes=classes, input_len=))
