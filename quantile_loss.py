import numpy as np


def QuantileLoss(perc, delta=1e-4):
    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = perc.reshape(1, -1)
    def _qloss(y, pred):
        I = tf.cast(y <= pred, tf.float32)
        error = K.abs(y - pred)
        correction = I * (1 - perc) + (1 - I) * perc
        # huber loss
        huber_loss = K.sum(correction * tf.where(error <= delta, 0.5 * error ** 2 / delta, error - 0.5 * delta), -1)
        print(huber_loss.shape)
        # order loss
        q_order_loss = K.sum(K.maximum(0.0, pred[:,:, :-1] - pred[:,:, 1:] + 1e-6), -1)
        print(q_order_loss.shape)
        return huber_loss + q_order_loss
    return _qloss






perc_points = [0.01, 0.25, 0.5, 0.75, 0.99]