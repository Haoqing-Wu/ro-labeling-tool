import glob
import sys
import scipy.io as sio
import numpy as np

TIME_STEP = 0.04  # Time span of MDF/ADTF signal scan
NR_OF_RAW_MAINPATH_POINTS = 40
NR_OF_CAMERA_ACTORS = 20
NO_ACTOR = 255.0

class MatLoader:
    def __init__(self, args):
        self.args = args
        self.ego_generator = EMLFromMat()
        self.actor_generator = OGMFromMat()
        self.signals = {}
        self.ego_paths = []
        self.actors_paths = []
        self.high = 0
        self.current = args.start_frame
        self.fpi = int(args.range / TIME_STEP)  # Frame per iteration: number of frames will be calculated in one iter
        file = glob.glob("{x}/*.{y}".format(x=self.args.data_folder, y='mat'))
        if len(file):
            print('Process data: {}'.format(file[0]))
        else:
            sys.exit('.mat not found')

        mat = sio.loadmat(file[0])
        self.flex_ray = mat['FlexRay']
        for k in self.flex_ray.dtype.fields.keys():
            if k == 'Time':
                self.signals['Time'] = self.flex_ray['Time'][0, 0]
                self.high = len(self.signals['Time'])
            else:
                obj = self.flex_ray[k][0, 0]
                for l in obj.dtype.fields.keys():
                    self.signals[l] = obj[l][0, 0]
        for key in list(self.signals.keys()):
            if not self.signals.get(key).any():
                self.signals.pop(key)
        self.signals_length = min([len(self.signals[i]) for i in self.signals.keys()])
        print('{} frames of signal are loaded'.format(self.signals_length))

    def generate_ego_paths(self):
        for idx in range(self.high):
            if self.current + self.fpi < self.high:
                mp_mock, compen_xy = compute_mock(self.make_signal_ego_path())
                self.ego_paths.append(mp_mock)
                self.current += 1
            else:
                self.current = 0
                break
        print("size of ego paths: {}".format(len(self.ego_paths)))

    def generate_actors_paths(self):

        self.actors_paths = self.make_signal_actor_path()
        print("size of actors paths: {}".format(len(self.actors_paths)))

    def collect_signal(self, idx, signals):
        found = self.signals.keys()
        frames = []
        for id, _type in signals:
            if id in found and idx <= len(self.signals[id]):
                frames.append(self.signals[id][idx][0])
            else:
                # signal is present in the ADTF mapping but not in the export file
                # or it has less frames than expected (requested). Either way, we use zero
                frames.append(0)
        return frames

    def make_signal_ego_path(self):
        start = self.current
        stop = self.current + self.fpi
        stop = min(stop, self.signals_length)
        ego_path_data = []
        for idx in range(start, stop):
            data = self.collect_signal(idx, self.ego_generator.signals)
            ego_path_data.append(data)

        path = self.ego_generator.compute_path(ego_path_data)
        return path

    def make_signal_actor_path(self):
        actors_path_data = []
        id_prev = self.collect_signal(0, self.actor_generator.signals)[0]
        cursor = 1
        while cursor < self.high:
            actor_path_data = []
            if id_prev == NO_ACTOR:
                data = self.collect_signal(cursor, self.actor_generator.signals)
                id_prev = data[0]
            else:
                for idx in range(cursor, self.high - 1):
                    data = self.collect_signal(idx, self.actor_generator.signals)
                    if data[0] == id_prev:
                        id_prev = data[0]
                        data.insert(0, idx)
                        actor_path_data.append(data)
                    else:
                        id_prev = data[0]
                        actors_path_data.append(actor_path_data)
                        cursor = idx
                        break
            cursor += 1

        return actors_path_data


class EMLFromMat:
    def __init__(self):
        """ Specify signals from mat file that computes main path mock """

        self.signals = [
            ('EML_PositionX'      , float), #0
            ('EML_PositionY'      , float), #1
            ('EML_Gierwinkel'     , float), #2
            ('EML_GeschwX'        , float), #3
            ('EML_BeschlX'        , float), #4
            ('EML_BeschlY'        , float), #5

        ]

        self.position_x = 0
        self.position_y = 0
        self.prev_x = 0
        self.prev_y = 0

    def compute_path(self, data):
        """
        Create main path mock, by taking from data a given number of samples
        :param data: EML signals
        :return: main path mock
        """

        # convert XY points to world coordinates, x_wc - x world coord
        x_wc = local2world(self.position_x, self.prev_x, data, 0) # 'EML_PositionX'
        y_wc = local2world(self.position_y, self.prev_y, data, 1) # 'EML_PositionY'

        self.prev_x, self.prev_y = data[0][0], data[0][1]
        self.position_x, self.position_y = x_wc[0], y_wc[0]

        # define a step to get signals from data, based on given number of samples
        samples = NR_OF_RAW_MAINPATH_POINTS
        data_len = len(data)
        if data_len > samples:
            step = int(data_len / samples) + 1
            data_end = min(int(step * samples), data_len)
        else:
            step = 1
            data_end = data_len

        main_path = []
        for data_idx in range(0, data_end, step):
           main_path.append(
               [
                   data_idx * 0.04,  # 'Time'
                   float(x_wc[data_idx]),  # 'EML_PositionX'
                   float(y_wc[data_idx]),  # 'EML_PositionY'
                   float(data[data_idx][2]),  # 'EML_YawAngle'
                   0,  # 'EML_Kurvature'
                   float(data[data_idx][3]),  # 'EML_VelocityX'
                   float(data[data_idx][4])  # 'EML_AccelerationX'
               ]
           )
        return main_path


class OGMFromMat:
    def __init__(self):
        self.signals = []
        self.signals.append(('BV2_Obj_01_ID', float))
        self.signals.append(('BV2_Obj_01_Klasse', float))
        self.signals.append(('BV2_Obj_01_PositionX', float))
        self.signals.append(('BV2_Obj_01_PositionY', float))
        self.signals.append(('BV2_Obj_01_GeschwX', float))
        self.signals.append(('BV2_Obj_01_GeschwY', float))


def local2world(position, prev_pos, data, idx_elem):
    """
    Compute absolute positions for EML on X,Y
    :param position:
    :param prev:
    :param data: segment data provided in USK
    :param idx_elem: 0 - EML X, 1 - EML Y
    :return:
    """
    position_world = []
    for idx in range(len(data)):
        cur_pos = data[idx][idx_elem]
        delta = cur_pos - prev_pos
        compensate = 0
        if delta < -8:
            compensate = 16
        elif delta > 8:
            compensate = -16
        position += (delta + compensate)
        position_world.append(position)
        prev_pos = cur_pos

    return position_world


def get_relative_coord(origin, points):
    """
    Get pose (x, y, yaw) relative to a given origin, yaw normalized to [-pi,+pi]

    The angle should be given in radians.
    :param points: vector of points
    :param origin: origin point
    :return: vector of points
    """
    ox, oy, oyaw = origin
    px, py, pyaw = np.array(points[0]), np.array(points[1]), np.array(points[2])

    # points coordinates relative to origin
    x_rel = np.cos(oyaw) * (px - ox) + np.sin(oyaw) * (py - oy)
    y_rel = -np.sin(oyaw) * (px - ox) + np.cos(oyaw) * (py - oy)
    yaw_rel = np.arctan2(np.sin(pyaw - oyaw), np.cos(pyaw - oyaw))  # limit to plus/minus pi

    x, y = get_global_coord([ox, oy, oyaw], [x_rel, y_rel])

    return x_rel, y_rel, yaw_rel, [x, y]


def get_global_coord(global_p, local_p):
    """

    :param origin:
    :param points:
    :return:
    """

    ox, oy, oyaw = global_p
    px, py  = np.array(local_p[0]), np.array(local_p[1])

    x_global = np.cos(oyaw) * px - np.sin(oyaw) * py + ox
    y_global = np.sin(oyaw) * px + np.cos(oyaw) * py + oy

    return x_global, y_global


def compute_mock(ego_groundtruth):
    """
    Convert to USK(ego) frame of reference
    1 - EML X, 2 - EML Y, 3 - EML YAW
    :param ego_groundtruth:
    :return:
    """
    # origin coord
    ox, oy, oyaw = ego_groundtruth[0][1], ego_groundtruth[0][2], ego_groundtruth[0][3]
    # get all the points from segment(segment represented by prediction horizon)
    x =   [ego_groundtruth[elem][1] for elem in range(len(ego_groundtruth))]
    y =   [ego_groundtruth[elem][2] for elem in range(len(ego_groundtruth))]
    yaw = [ego_groundtruth[elem][3] for elem in range(len(ego_groundtruth))]

    # compen_xy represent the global coordinates computed from relative coordinates
    x_rel, y_rel, yaw_rel, compen_xy = get_relative_coord([ox, oy, oyaw], [x, y, yaw])

    mp_mock = ego_groundtruth
    for idx in range(ego_groundtruth.__len__()):
        mp_mock[idx][1] = x_rel[idx]
        mp_mock[idx][2] = y_rel[idx]
        mp_mock[idx][3] = yaw_rel[idx]
        mp_mock[idx].append(0.0)  # dtrack - covered distance at node; unit [m] - not used at the moment

    return mp_mock, compen_xy
