import tensorflow as tf

def placeholder(I, J, N):
    '''
    x:       [None, I, J, N]
    Time:    [None, N + 1, 3]
    Weather: [None, N + 1, 4]
    label:   [None, I, J, 1]
    '''
    x = tf.compat.v1.placeholder(shape = (None, I, J, N), dtype = tf.float32, name = 'x')
    Time = tf.compat.v1.placeholder(shape = (None, N + 1, 3), dtype = tf.int32, name = 'Time')
    Weather = tf.compat.v1.placeholder(shape = (None, N + 1, 4), dtype = tf.float32, name = 'Weather')
    label = tf.compat.v1.placeholder(shape = (None, I, J), dtype = tf.float32, name = 'label')
    return x, Time, Weather, label

def FC(x, d1, d2):
    x = tf.keras.layers.Dense(units = d1, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(rate = 0.2)(x)
    x = tf.keras.layers.Dense(units = d2, activation = None)(x)
    return x

def Conv3d(x, num_3dcnn, d):
    for _ in range(num_3dcnn - 1):
        x = tf.keras.layers.Conv3D(
            filters = d, kernel_size = (3, 3, 3), padding = 'same', activation = 'relu')(x)
        x = tf.keras.layers.Dropout(rate = 0.2)(x)
    x = tf.keras.layers.Conv3D(
        filters = 1, kernel_size = (3, 3, 3), padding = 'same', activation = None)(x)
    x = tf.squeeze(x, axis = -1)
    x = tf.transpose(x, perm = (0, 2, 3, 1))
    return x

def ResNet(x, num_res, d):
    x1 = tf.keras.layers.Conv2D(
        filters = d, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
    for _ in range(num_res):
        x = tf.keras.layers.Dropout(rate = 0.2)(x1)
        x = tf.keras.layers.Conv2D(
            filters = d, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
        x = tf.keras.layers.Dropout(rate = 0.2)(x)
        x = tf.keras.layers.Conv2D(
            filters = d, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
        x1 = tf.add(x1, x)
    x1 = tf.keras.layers.Dropout(rate = 0.2)(x1)
    x = tf.keras.layers.Conv2D(
        filters = 1, kernel_size = (3, 3), padding = 'same', activation = None)(x1)
    return x

def model(x, Time, Weather, IIF, T, num_WC, num_3dcnn, num_res, d):
    batch_size = tf.shape(x)[0]
    I = x.get_shape()[1].value
    J = x.get_shape()[2].value
    N = x.get_shape()[3].value
    K = IIF.shape[-1]
    # IIF
    IIF = tf.expand_dims(tf.expand_dims(IIF, axis = 0), axis = 0)
    IIF = tf.tile(IIF, multiples = (batch_size, N + 1, 1, 1, 1))
    # DIF
    dayofweek = tf.one_hot(Time[..., 0], depth = 7)
    timeofday = tf.one_hot(Time[..., 1], depth = T)
    daytype = tf.one_hot(Time[..., 2], depth = 2)
    Time = tf.concat((dayofweek, timeofday, daytype), axis = -1)
    Time = FC(Time, d, K)
    WN = Weather[..., : -1]
    WC = tf.cast(Weather[..., -1], dtype = tf.int32)
    WC = tf.one_hot(WC, depth = num_WC)
    DIF = tf.concat((Time, WN, WC), axis = -1)
    DIF = FC(DIF, d, K)
    DIF = tf.expand_dims(tf.expand_dims(DIF, axis = 2), axis = 2)
    DIF = tf.tile(DIF, multiples = (1, 1, I, J, 1))
    # STD
    IDIF = tf.multiply(IIF, DIF)
    STD = Conv3d(IDIF, num_3dcnn, d)
    # prediction
    x = x - STD[..., : -1]
    x = ResNet(x, num_res, d)
    x = x + STD[..., -1 :]
    return tf.squeeze(x, axis = -1)
    
def mse_loss(pred, label):
    loss = (pred - label) ** 2
    loss = tf.reduce_mean(loss)
    return loss  
    
