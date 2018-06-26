import sys
sys.path.append('/home/xiaocc/Documents/caffe-cpm/python')
import caffe
from caffe.proto import caffe_pb2
import data_transformer
import numpy as np

class CPMDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.batch_size = params['batch_size']
        self.batch_loader = data_transformer.BatchLoader(params, None)
        top[0].reshape(self.batch_size, 3, params['crop_size_x'], params['crop_size_y'])
        g_x = params['crop_size_x'] / params['stride']
        g_y = params['crop_size_y'] / params['stride']
        top[1].reshape(self.batch_size, params['num_parts']+1, g_x, g_y)

        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            img, label = self.batch_loader.load_next_image_label()

            in_ = np.array(img, dtype=np.float32)
            in_ = in_ / 256.0 - 0.5
            in_ = in_.transpose((2, 0, 1))

            lb_ = np.array(label, dtype=np.float32)
            lb_ = lb_.transpose((2, 0, 1))
            top[0].data[itt, ...] = in_
            top[1].data[itt, ...] = lb_

    def backward(self, top, propagate_down, bottom):
        pass



