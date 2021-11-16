import numpy as np
import pandas as pd

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, truth):
    RMSE = np.sqrt(np.mean((pred - truth) ** 2))
    return RMSE

def seq2instance(inputs, num_his, num_pred = 1, dtype = np.float32):
    num_instance = inputs.shape[0] - num_his - num_pred + 1
    x = np.zeros(shape = (num_instance, num_his) + inputs.shape[1 :], dtype = dtype)
    y = np.zeros(shape = (num_instance, num_pred) + inputs.shape[1 :], dtype = dtype)
    for i in range(num_instance):
        x[i] = inputs[i : i + num_his]
        y[i] = inputs[i + num_his : i + num_his + num_pred]
    return x, y

def load_data(args):
    # x
    df = pd.read_hdf(args.data_file)
    x = df.values.astype(np.float32)
    # IIF
    IIF = np.load(args.IIF_file).astype(np.float32)
    I, J, K = IIF.shape
    minimum, maximum = np.min(IIF), np.max(IIF)
    IIF = (IIF - minimum) / (maximum - minimum)
    # context
    num_train = int(args.train_day * 24 * 60 / args.time_interval - args.N)
    num_test = int(args.test_day * 24 * 60 / args.time_interval + args.N)
    num_val = int(0.2 * num_train)
    num_train -= num_val
    Time = df.index
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) // (args.time_interval * 60)
    timeofday = np.reshape(timeofday, newshape = (-1, 1))  
    daytype = np.logical_or(dayofweek == 5, dayofweek == 6) # 0: workday, 1: weekend
    Time = np.concatenate((dayofweek, timeofday, daytype), axis = -1)
    Weather = np.load(args.Weather_file) # temperature, visibility, wind speed, weather condition
    minimum, maximum = np.min(Weather[: -num_test, : -1], axis = 0), np.max(Weather[: -num_test, : -1], axis = 0)
    Weather[:, : -1] = (Weather[:, : -1] - minimum) / (maximum - minimum)
    # train/val/test
    x = np.reshape(x, newshape = (-1, I, J))
    mean, std = np.mean(x[: -num_test]), np.std(x[: -num_test])
    x = (x - mean) / std
    train_x, train_y = seq2instance(x[: num_train], args.N, 1, np.float32)
    train_x = np.transpose(train_x, axes = (0, 2, 3, 1))
    train_y = np.transpose(train_y, axes = (0, 2, 3, 1))
    val_x, val_y = seq2instance(x[num_train : num_train + num_val], args.N, 1, np.float32)
    val_x = np.transpose(val_x, axes = (0, 2, 3, 1))
    val_y = np.transpose(val_y, axes = (0, 2, 3, 1))    
    test_x, test_y = seq2instance(x[-num_test :], args.N, 1, np.float32)
    test_x = np.transpose(test_x, axes = (0, 2, 3, 1))
    test_y = np.transpose(test_y, axes = (0, 2, 3, 1))
    train_Time = seq2instance(Time[: num_train], args.N, 1, np.int32)
    train_Time = np.concatenate(train_Time, axis = 1)
    val_Time = seq2instance(Time[num_train : num_train + num_val], args.N, 1, np.int32)
    val_Time = np.concatenate(val_Time, axis = 1)
    test_Time = seq2instance(Time[-num_test :], args.N, 1, np.int32)
    test_Time = np.concatenate(test_Time, axis = 1)
    train_Weather = seq2instance(Weather[: num_train], args.N, 1, np.float32)
    train_Weather = np.concatenate(train_Weather, axis = 1)
    val_Weather = seq2instance(Weather[num_train : num_train + num_val], args.N, 1, np.float32)
    val_Weather = np.concatenate(val_Weather, axis = 1)
    test_Weather = seq2instance(Weather[-num_test :], args.N, 1, np.float32)
    test_Weather = np.concatenate(test_Weather, axis = 1)
    return (train_x, train_Time, train_Weather, train_y[..., 0], val_x, val_Time, val_Weather, val_y[..., 0],
            test_x, test_Time, test_Weather, test_y[..., 0], IIF, mean, std)

