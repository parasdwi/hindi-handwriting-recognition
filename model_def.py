import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_H, IMG_W = 32, 128


class BilinearSampler(layers.Layer):

    def call(self, inputs):
        img, theta = inputs
        B = tf.shape(img)[0]
        H, W, C = img.shape[1], img.shape[2], img.shape[3]
        theta = tf.reshape(theta, (B, 2, 3))

        xs = tf.linspace(-1.0, 1.0, W)
        ys = tf.linspace(-1.0, 1.0, H)
        xg, yg = tf.meshgrid(xs, ys)
        grid = tf.stack([tf.reshape(xg, [-1]), tf.reshape(yg, [-1]),
                         tf.ones([H * W])], axis=0)
        grid = tf.tile(grid[None], [B, 1, 1])

        src = tf.matmul(theta, grid)
        x = (src[:, 0, :] + 1.0) * 0.5 * (W - 1.0)
        y = (src[:, 1, :] + 1.0) * 0.5 * (H - 1.0)

        x0 = tf.floor(x)
        y0 = tf.floor(y)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        flat = tf.reshape(img, (B, H * W, C))

        def corner(yi, xi, weight):
            yi = tf.cast(yi, tf.int32)
            xi = tf.cast(xi, tf.int32)
            inside = tf.cast((xi >= 0) & (xi <= W - 1) & (yi >= 0) & (yi <= H - 1), tf.float32)
            idx = tf.clip_by_value(yi, 0, H - 1) * W + tf.clip_by_value(xi, 0, W - 1)
            vals = tf.gather(flat, idx, batch_dims=1)
            return vals * (weight * inside)[..., None]

        out = (corner(y0, x0, (x1 - x) * (y1 - y)) +
               corner(y1, x0, (x1 - x) * (y - y0)) +
               corner(y0, x1, (x - x0) * (y1 - y)) +
               corner(y1, x1, (x - x0) * (y - y0)))
        return tf.reshape(out, (B, H, W, C))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class SelfAttention(layers.Layer):

    def __init__(self, units=512, key_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.units, self.key_dim = units, key_dim
        self.q_proj = layers.Dense(key_dim)
        self.k_proj = layers.Dense(key_dim)
        self.v_proj = layers.Dense(units)

    def call(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.sqrt(tf.cast(self.key_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, v)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)


def identity_theta_bias(shape, dtype=None):
    return tf.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=dtype)


def conv(x, filters, k, bn=False, pool=None):
    x = layers.Conv2D(filters, k, padding="same", activation="relu")(x)
    if bn:
        x = layers.BatchNormalization()(x)
    if pool:
        x = layers.MaxPool2D(pool_size=pool)(x)
    return x


def build_scrat_net(num_classes, use_residual=True):
    inp = keras.Input(shape=(IMG_H, IMG_W, 1), name="image")

    s = layers.Conv2D(16, 5, strides=2, padding="same", activation="relu")(inp)
    s = layers.MaxPool2D(2)(s)
    s = layers.Conv2D(32, 5, strides=2, padding="same", activation="relu")(s)
    s = layers.MaxPool2D(2)(s)
    s = layers.Flatten()(s)
    s = layers.Dense(120, activation="relu")(s)
    s = layers.Dropout(0.2)(s)
    s = layers.Dense(84, activation="relu")(s)
    theta = layers.Dense(6, kernel_initializer="zeros",
                         bias_initializer=identity_theta_bias, name="theta")(s)
    x = BilinearSampler(name="stn_warp")([inp, theta])

    x = conv(x, 32, 5, pool=(2, 2))
    x = conv(x, 64, 5, pool=(2, 1))
    x = conv(x, 128, 3)
    x = conv(x, 128, 3, pool=(2, 1))
    x = conv(x, 256, 3, bn=True)
    x = conv(x, 512, 3, pool=(2, 1))
    x = conv(x, 512, 2, bn=True)
    x = conv(x, 512, 2, bn=True)
    x = conv(x, 512, 2, bn=True, pool=(2, 1))
    x = conv(x, 512, 1)
    x = layers.Reshape((x.shape[2], x.shape[3]))(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)

    att = SelfAttention(units=512, key_dim=64, name="self_attention")(x)
    x = layers.Add()([x, att]) if use_residual else att

    out = layers.Dense(num_classes, activation="softmax", name="char_probs")(x)
    return keras.Model(inp, out, name="scrat_net")
