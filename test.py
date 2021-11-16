import time
import math
import utils
import model
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--time_interval', default = 15, type = int, help = 'a time step is 15 mins')
parser.add_argument('--N', default = 4, type = int, help = 'number of historical time steps')
parser.add_argument('--batch_size', default = 32, type = int, help = 'batch size')
parser.add_argument('--train_day', default = 23, type = int, help = '23 days for training')
parser.add_argument('--test_day', default = 7, type = int, help = '7 days for testing')
parser.add_argument('--data_file', default = './data/Chengdu.h5', type = str)
parser.add_argument('--IIF_file', default = './data/IIF.npy', type = str)
parser.add_argument('--Weather_file', default = './data/Weather.npy', type = str)
parser.add_argument('--model_file', default = './data/DeepSTD', type = str)
parser.add_argument('--log_file', default = './data/log', type = str)
args = parser.parse_args()
log = open(args.log_file, 'w')

# load data
start = time.time()
utils.log_string(log, 'loading data...')
data = utils.load_data(args)
test_x, test_Time, test_Weather, test_y, _, mean, std = data[8 :]
utils.log_string(log, 'data loaded!')

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
graph = tf.Graph()
with graph.as_default():
    saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph = graph, config = config) as sess:
    saver.restore(sess, args.model_file)
    parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
    pred = graph.get_collection(name = 'pred')[0]
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')
    test_pred = []
    num_test = test_x.shape[0]
    num_batch = math.ceil(num_test / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, num_test)        
        feed_dict = {
            'x:0': test_x[start_idx : end_idx],
            'Time:0': test_Time[start_idx : end_idx],
            'Weather:0': test_Weather[start_idx : end_idx]}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        test_pred.append(pred_batch)
    test_pred = np.concatenate(test_pred, axis = 0)
# metric
test_pred = test_pred * std + mean
test_y = test_y * std + mean
workday = test_Time[:, -1, -1] == 0
weekend = test_Time[:, -1, -1] == 1
rmse1 = utils.metric(test_pred[workday], test_y[workday])
rmse2 = utils.metric(test_pred[weekend], test_y[weekend])
rmse3 = utils.metric(test_pred, test_y)
utils.log_string(log, 'workday: %.2f\tweekend: %.2f\tall day: %.2f' % (rmse1, rmse2, rmse3))
end = time.time()
utils.log_string(log, 'total time: %.2fs' % (end - start))
log.close()
sess.close()
