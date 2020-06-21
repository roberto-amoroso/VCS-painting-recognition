from __future__ import division
from yolo.util import *
from yolo.darknet import Darknet
import random
import pickle as pkl


class PeopleDetection():

    def __init__(self):
        super().__init__()
        self.cfgfile = "yolo/cfg/yolov3.cfg"
        self.weightsfile = "yolo/yolov3.weights"
        self.num_classes = 80
        self.confidence = 0.25
        self.nms_thesh = 0.4
        self.reso = 160

        self.start = 0
        self.CUDA = torch.cuda.is_available()

        self.num_classes = 80
        self.bbox_attrs = 5 + self.num_classes

        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weightsfile)

        self.model.net_info["height"] = self.reso
        self.inp_dim = int(self.model.net_info["height"])

        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        if self.CUDA:
            self.model.cuda()

        # Switch to “evaluate” mode before predictions
        self.model.eval()

        self.classes = load_classes('yolo/data/coco.names')
        self.colors = pkl.load(open("yolo/pallete", "rb"))

    def prep_image(self, img, inp_dim):
        """
        Prepare image for inputting to the neural network. 

        Returns a Variable 
        """

        orig_im = img
        dim = orig_im.shape[1], orig_im.shape[0]
        img = cv2.resize(orig_im, (inp_dim, inp_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim

    def write(self, x, img, people_bounding_boxes, only_person=False):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        cls = int(x[-1])
        if (not only_person) or (only_person and self.classes[cls] == 'person'):
            label = "{0}".format(self.classes[cls])
            color = random.choice(self.colors)
            cv2.rectangle(img, c1, c2, color, 2)

            x = int(c1[0])
            y = int(c1[1])

            w = int(c2[0] - x)
            h = int(c2[1] - y)

            people_bounding_boxes.append([x, y, w, h])

            font_scale = 1.5
            line_thickness = 2
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, line_thickness)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, font_scale, [225, 255, 255], line_thickness)
        return img

    def run(self, frame):
        img, orig_im, dim = self.prep_image(frame, self.inp_dim)

        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if self.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        output = self.model(Variable(img), self.CUDA)
        output = write_results(output, self.confidence, self.num_classes,
                               nms=True, nms_conf=self.nms_thesh)

        if type(output) == int:
            return orig_im, False, []

        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(self.inp_dim)) / self.inp_dim

        #            im_dim = im_dim.repeat(output.size(0), 1)
        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        people_bounding_boxes = []

        list(map(lambda x: self.write(x, orig_im, people_bounding_boxes, only_person=True), output))

        return orig_im, True, people_bounding_boxes
