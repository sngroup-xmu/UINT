import json
import random
import numpy as np
import tensorflow.compat.v1 as tf

# import torch
def relu(inX):
    return np.maximum(0, inX)

class NeurouNetwork():

    def __init__(self, path, system):

        # if system == 'demo':
        #     self._weights, self._biases, self._activ = self._parse_demo(path)
        # elif 'aurora' in system:
        #     self._weights, self._biases, self._activ = self._parse_aurora(path)
        # elif system == 'pensieve':
        #     self._weights, self._biases, self._activ = self._parse_pensieve(path)
        
        if system == 'aurora':
            self._weights, self._biases, self._activ = self._parse_aurora(path)
        elif system == 'pensieve':
            self._weights, self._biases, self._activ = self._parse_pensieve(path)
        
    # def _parse_demo(self, path):
    #     with tf.io.gfile.GFile(path, "rb") as f:
    #         graph_def = tf.GraphDef()
    #         # Get graph info from '.pb' file
    #         graph_def.ParseFromString(f.read())

    #         # Get weights and biases by names
    #         elem_list = ["w1:0", "b1:0", "w2:0", "b2:0"]
    #         w1, b1, w2, b2 = tf.import_graph_def(graph_def, return_elements=elem_list)
    #         w1 = w1.eval(session=tf.Session())
    #         b1 = b1.eval(session=tf.Session())
    #         w2 = w2.eval(session=tf.Session())
    #         b2 = b2.eval(session=tf.Session())

    #         weights = [w1, w2]
    #         biases = [b1, b2]
    #         # print(weights,biases)
    #         return weights, biases, 'relu'

    def _parse_aurora(self, path):
        with tf.gfile.FastGFile(path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            elem_list = ["model/pi_fc0/w/read:0", "model/pi_fc0/b/read:0",
                         "model/pi_fc1/w/read:0", "model/pi_fc1/b/read:0",
                         "model/pi/w/read:0", "model/pi/b/read:0"]
            w1, b1, w2, b2, w3, b3 = tf.import_graph_def(graph_def, return_elements=elem_list)
            w1 = w1.eval(session=tf.Session())
            b1 = b1.eval(session=tf.Session())
            w2 = w2.eval(session=tf.Session())
            b2 = b2.eval(session=tf.Session())
            w3 = w3.eval(session=tf.Session())
            b3 = b3.eval(session=tf.Session())

            weights = [w1, w2, w3]
            biases = [b1, b2, b3]
            # print(weights,biases)
            return weights, biases, 'relu'

    def _parse_pensieve(self, path):
        with tf.gfile.FastGFile(path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            # Get first layer parameters
            elem_list = [
                "actor/FullyConnected/W/read:0", "actor/FullyConnected/b/read:0",
                "actor/FullyConnected_1/W/read:0", "actor/FullyConnected_1/b/read:0",
                "actor/Conv1D/W/read:0", "actor/Conv1D/b/read:0",
                "actor/Conv1D_1/W/read:0", "actor/Conv1D_1/b/read:0",
                "actor/Conv1D_2/W/read:0", "actor/Conv1D_2/b/read:0",
                "actor/FullyConnected_2/W/read:0", "actor/FullyConnected_2/b/read:0",
            ]
            prev_bit_w, prev_bit_b, buffer_w, buffer_b, \
            throughput_w, throughput_b, latency_w, latency_b, \
            next_sz_w, next_sz_b, remain_trunk_w, remain_trunk_b = \
                tf.import_graph_def(graph_def, return_elements=elem_list)

            prev_bit_w = prev_bit_w.eval(session=tf.Session())
            prev_bit_b = prev_bit_b.eval(session=tf.Session())

            buffer_w = buffer_w.eval(session=tf.Session())
            buffer_b = buffer_b.eval(session=tf.Session())

            throughput_w = throughput_w.eval(session=tf.Session())[1, 0]
            throughput_b = throughput_b.eval(session=tf.Session())

            latency_w = latency_w.eval(session=tf.Session())[1, 0]
            latency_b = latency_b.eval(session=tf.Session())

            next_sz_w = next_sz_w.eval(session=tf.Session())[1, 0]
            next_sz_b = next_sz_b.eval(session=tf.Session())

            remain_trunk_w = remain_trunk_w.eval(session=tf.Session())
            remain_trunk_b = remain_trunk_b.eval(session=tf.Session())

            # prev_bit_w = np.concatenate((prev_bit_w, np.zeros((1, 128 * 5))), axis=1)
            # buffer_w = np.concatenate((np.zeros((1, 128)), buffer_w, np.zeros((1, 128 * 4))), axis=1)
            # throughput_w = np.concatenate((np.zeros((throughput_w.shape[0], 128 * 2)), throughput_w, np.zeros((throughput_w.shape[0], 128 * 3))), axis=1)
            # latency_w = np.concatenate((np.zeros((latency_w.shape[0], 128 * 3)), latency_w, np.zeros((latency_w.shape[0], 128 * 2))), axis=1)
            # next_sz_w = np.concatenate((np.zeros((next_sz_w.shape[0], 128 * 4)), next_sz_w, np.zeros((next_sz_w.shape[0], 128))), axis=1)
            # remain_trunk_w = np.concatenate((np.zeros((1, 128 * 5)), remain_trunk_w), axis=1)
            #
            # w1 = np.concatenate((prev_bit_w, buffer_w, throughput_w, latency_w, next_sz_w, remain_trunk_w))
            # b1 = np.concatenate((prev_bit_b, buffer_b, throughput_b, latency_b, next_sz_b, remain_trunk_b))

            w1 = [prev_bit_w, buffer_w, throughput_w, latency_w, next_sz_w, remain_trunk_w]
            b1 = [prev_bit_b, buffer_b, throughput_b, latency_b, next_sz_b, remain_trunk_b]

            # Hidden layers and output layer parameters
            elem_list = [
                "actor/FullyConnected_3/W/read:0", "actor/FullyConnected_3/b/read:0",
                "actor/FullyConnected_4/W/read:0", "actor/FullyConnected_4/b/read:0"
            ]
            w2, b2, w3, b3 = tf.import_graph_def(graph_def, return_elements=elem_list)
            w2 = w2.eval(session=tf.Session())
            b2 = b2.eval(session=tf.Session())
            w3 = w3.eval(session=tf.Session())
            b3 = b3.eval(session=tf.Session())

            weights, biases = [w1, w2, w3], [b1, b2, b3]

            return weights, biases, 'relu'
    
    def _parse_confuciu(self, path):
        model_dict = torch.load(path, map_location=torch.device('cpu'))
    
        w = []
        b = []
        name = []
    
        for key in model_dict.keys():
            name.append(key)
            if 'weight' in key:
                w.append(np.array(model_dict[key]))
            elif 'bias' in key:
                b.append(np.array(model_dict[key]))
        # print(w,b)
        return  w, b,'relu'

    # Simulate neurou network forward propagation process
    def forward(self, inputs):
        prob = inputs
        for i in range(len(self._weights) - 1):
            weight = self._weights[i]
            if isinstance(weight, list):
                tmp = []
                pre_len = 0
                for j in range(len(weight)):
                    w = weight[j]
                    length = w.shape[0]
                    tmp.extend(np.dot(prob[pre_len: pre_len + length], w) + self._biases[i][j])
                    pre_len += length
                prob = relu(tmp)
            else:
                prob = relu(np.dot(prob, self._weights[i]) + self._biases[i]) #矩阵乘法
        prob = np.dot(prob, self._weights[-1]) + self._biases[-1]
        return prob


    def parameters(self):
        return self._weights, self._biases, self._activ

if __name__ == '__main__':

    nn = NeurouNetwork('../../model_file/demo/demo.pb', 'demo')

    dataset = json.load(open('../../model_file/demo/demo_dataset.json', 'r'))

    inputs = dataset['data'][0]['x']

    print(nn.forward(inputs))
