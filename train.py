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
parser.add_argument('--num_3dcnn', default = 1, type = int, help = 'number of 3D CNN layers')
parser.add_argument('--num_res', default = 6, type = int, help = 'number of residual units')
parser.add_argument('--d', default = 64, type = int, help = 'number of filters')
parser.add_argument('--batch_size', default = 32, type = int, help = 'batch size')
parser.add_argument('--lr', default = 0.001, type = float, help = 'learning rate')
parser.add_argument('--epochs', default = 1000, type = int, help = 'max epoch')
parser.add_argument('--patience', default = 30, type = int, help = 'early stop')
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
(train_x, train_Time, train_Weather, train_y, val_x, val_Time, val_Weather, val_y,
 test_x, test_Time, test_Weather, test_y, IIF, mean, std) = utils.load_data(args)
utils.log_string(log, 'train_x: %s\ttrain_y: %s' % (train_x.shape, train_y.shape))
utils.log_string(log, 'val_x:   %s\tval_y:   %s' % (val_x.shape, val_y.shape))
utils.log_string(log, 'test_x:  %s\ttest_y:  %s' % (test_x.shape, test_y.shape))
utils.log_string(log, 'data loaded!')

# train model
utils.log_string(log, 'compling model...')
I, J, _ = IIF.shape
T = int(24 * 60 / args.time_interval)
num_WC = np.max(train_Weather[..., -1]) + 1
x, Time, Weather, label = model.placeholder(I, J, args.N)
pred = model.model(x, Time, Weather, IIF, T, num_WC, args.num_3dcnn, args.num_res, args.d)
loss = model.mse_loss(pred, label)
tf.compat.v1.add_to_collection('pred', pred)
tf.compat.v1.add_to_collection('loss', loss)
optimizer = tf.compat.v1.train.AdamOptimizer(args.lr)
global_step = tf.Variable(0, trainable = False)
train_op = optimizer.minimize(loss, global_step = global_step)
parameters = 0
for variable in tf.compat.v1.trainable_variables():
    parameters += np.product([x.value for x in variable.get_shape()])
utils.log_string(log, 'total trainable parameters: {:,}'.format(parameters))
utils.log_string(log, 'model compiled!')
saver = tf.compat.v1.train.Saver()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
sess.run(tf.compat.v1.global_variables_initializer())
utils.log_string(log, '**** training model ****')
train_time, val_time = [], []
val_loss_min = np.inf
wait = 0
num_train = train_x.shape[0]
num_val = val_x.shape[0]
num_test = test_x.shape[0]
for epoch in range(args.epochs):
    if wait >= args.patience:
        print('early stop at epoch: %d' % (epoch))
        break
    # train loss
    permutation = np.random.permutation(num_train)
    train_x = train_x[permutation]
    train_Time = train_Time[permutation]
    train_Weather = train_Weather[permutation]
    train_y = train_y[permutation]
    train_loss = []
    num_batch = math.ceil(num_train / args.batch_size)
    t1 = time.time()
    for i in range(num_batch):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, num_train)
        feed_dict = {
            x: train_x[start_idx : end_idx],
            Time: train_Time[start_idx : end_idx],
            Weather: train_Weather[start_idx : end_idx],
            label: train_y[start_idx : end_idx]}
        _, loss_batch = sess.run([train_op, loss], feed_dict = feed_dict)
        train_loss.append(loss_batch)
    t2 = time.time()
    train_time.append(t2 - t1)
    train_loss = np.mean(train_loss)     
    # val loss
    val_loss = []
    num_batch = math.ceil(num_val / args.batch_size)
    t1 = time.time()
    for i in range(num_batch):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, num_val)        
        feed_dict = {
            x: val_x[start_idx : end_idx],
            Time: val_Time[start_idx : end_idx],
            Weather: val_Weather[start_idx : end_idx],
            label: val_y[start_idx : end_idx]}
        loss_batch = sess.run(loss, feed_dict = feed_dict)
        val_loss.append(loss_batch)
    t2 = time.time()
    val_time.append(t2 - t1)
    val_loss = np.mean(val_loss)
    utils.log_string(
        log, '%s | epoch: %03d/%d, train time: %.2fs, train loss: %.5f, val time: %.2fs, val loss: %.5f' %
        (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, args.epochs, train_time[-1], train_loss, val_time[-1], val_loss))
    if val_loss <= val_loss_min:
        utils.log_string(log, 'val loss decrease from %.5f to %.5f, saving model to %s' % (val_loss_min, val_loss, args.model_file))
        val_loss_min = val_loss
        saver.save(sess, args.model_file)
        wait = 0
    else:
        wait += 1
utils.log_string(
    log, 'train finish, ave train time: %.2fs, ave val time: %.2fs, min val loss: %.5f' %
    (np.mean(train_time), np.mean(val_time), val_loss_min))

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
saver.restore(sess, args.model_file)
utils.log_string(log, 'model restored!')
utils.log_string(log, 'evaluating...')
test_pred = []
num_batch = math.ceil(num_test / args.batch_size)
for i in range(num_batch):
    start_idx = i * args.batch_size
    end_idx = min((i + 1) * args.batch_size, num_test)        
    feed_dict = {
        x: test_x[start_idx : end_idx],
        Time: test_Time[start_idx : end_idx],
        Weather: test_Weather[start_idx : end_idx]}
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
utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
log.close()
sess.close()
