import glob
import sys
import scipy.io as sio


class MatLoader:
    def __init__(self, args):
        self.args = args
        self.signals = {}
        self.high = 0
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