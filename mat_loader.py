import glob
import itertools
import sys
import scipy.io as sio
import utils
from tqdm import tqdm


TIME_STEP = 0.04  # Time span of MDF/ADTF signal scan
NR_OF_RAW_MAINPATH_POINTS = 40
NR_OF_CAMERA_ACTORS = 10
NR_OF_LRR_ACTORS = 20
NO_ACTOR = 255.0 # No actor is present then the ID from sensor will be 255
FRAGMENT_LENGTH = 4.0  # [s] length of the fragment which will be labeled

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

        print("Generate Ego trajectories for each frame.")
        for idx in tqdm(range(self.high - self.fpi)):
            mp_mock, compen_xy = utils.compute_mock(self.make_signal_ego_path())
            self.ego_paths.append(mp_mock)

    def generate_actors_paths(self):

        print("Generate Actors trajectories from BV2.")
        paths = []
        for object_id in tqdm(range(NR_OF_CAMERA_ACTORS)):
            paths.append(self.make_signal_actor_path(object_id, 'BV2'))
        self.actors_paths = list(itertools.chain.from_iterable(paths))
        print("Size of BV2 actors path fragment: {}".format(len(self.actors_paths)))

    def collect_ego_signal(self, idx, signals):
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
    
    def collect_actor_signal(self, idx, signals, object_id):
        """
        Collects the actor data from the mat file and returns a list of paths
        :param idx: index of the frame
        :param signals: list of signals
        :param object_id: id of the actor
        :return: list of paths
        """
        found = self.signals.keys()
        frames = []
        for id, _type in signals[object_id]:
            if id in found and idx < len(self.signals[id]):
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
            data = self.collect_ego_signal(idx, self.ego_generator.signals)
            ego_path_data.append(data)

        path = self.ego_generator.compute_path(ego_path_data)
        return path

    def make_signal_actor_path(self, object_id, sensor):
        """
        Collects the actor data from the mat file and returns a list of paths
        :param object_id: id of the actor
        :return: list of paths
        """
        actors_path_data = []
        
        mask = self.actor_generator.signals[sensor]
        id_prev = self.collect_actor_signal(0, mask, object_id)[0]
        cursor = 1
        while cursor < self.high:
            actor_path_data = []
            if id_prev == NO_ACTOR:
                data = self.collect_actor_signal(cursor, mask, object_id) # get the next actor
                id_prev = data[0]
            else:
                for idx in range(cursor, self.high - 1):
                    data = self.collect_actor_signal(idx, mask, object_id)
                    if data[0] == id_prev: # same actor
                        id_prev = data[0]
                        data.insert(0, idx)
                        actor_path_data.append(data)
                        # if the length of the fragment is reached then truncate the path
                        if len(actor_path_data) >= FRAGMENT_LENGTH / TIME_STEP: # 4 seconds , 100 frames
                            actors_path_data.append(actor_path_data)
                            cursor = idx
                            break
                    else: # new actor
                        id_prev = data[0] # update id
                        # actors_path_data.append(actor_path_data)
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
        x_wc = utils.local2world(self.position_x, self.prev_x, data, 0) # 'EML_PositionX'
        y_wc = utils.local2world(self.position_y, self.prev_y, data, 1) # 'EML_PositionY'

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
        self.bv2_signals = [[
            ('BV2_Obj_{:0>2d}_ID'.format(i), float),
            ('BV2_Obj_{:0>2d}_Klasse'.format(i), float),
            ('BV2_Obj_{:0>2d}_PositionX'.format(i), float),
            ('BV2_Obj_{:0>2d}_PositionY'.format(i), float),
            ('BV2_Obj_{:0>2d}_GeschwX'.format(i), float),
            ('BV2_Obj_{:0>2d}_GeschwY'.format(i), float),
        ] for i in range(1, 11)]
        
        self.LRR1_signals = [[
            ('LRR1_Obj_{:0>2d}_ID_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_Klasse_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_RadialDist_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_AzimutWnkl_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_GierWnkl_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_RadialGeschw_UF'.format(i), float),
        ] for i in range(1, 21)]

        self.signals = {
            'BV2': self.bv2_signals,
            'LRR1': self.LRR1_signals
        }


