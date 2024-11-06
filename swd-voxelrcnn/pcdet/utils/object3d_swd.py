import numpy as np
import json

def get_objects_from_label(row):
    objects = [Object3d(row)]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'steering_wheel': 1}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, row):
        self.src = row
        self.cls_type = 'steering_wheel'
        self.cls_id = cls_type_to_id(self.cls_type)
        self.l = 0.385
        self.w = 0.385
        self.h = 4 * 0.035 # in fact 0.035
        self.loc = np.fromstring(row.sw_translation_vector[1:-1], sep=' ', dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        angles = np.fromstring(row.sw_angles[1:-1], sep=' ', dtype=np.float32)
        angles = np.radians(angles)
        direct_cosines = np.cos(angles)
        self.rx = - float(np.arctan2(direct_cosines[1], direct_cosines[2]))
        #self.ry = float(angles[1])
        #self.rz = float(angles[2])
        #rotation_matrix = np.fromstring(row.sw_rotation_matrix.replace('[', '').replace(']', ''), sep=' ', dtype=np.float32)
        #self.R = rotation_matrix.reshape(3,3)
        self.level = 1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        """
        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        """
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(self.R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str
