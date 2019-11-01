import tensorflow as tf

def p_to_tensor(p, shape, dtype=tf.float32, seed=1337):
    if type(p) == tuple and p[0] != p[1]:
        if dtype == tf.uint8:
            return tf.cast(tf.random.uniform(shape=shape, dtype=tf.int32, minval=p[0], maxval=p[1], seed=seed), tf.uint8)
        return tf.random.uniform(shape=shape, dtype=dtype, minval=p[0], maxval=p[1], seed=seed)
    elif type(p) == list and len(p) != 1:
        logits = tf.random.uniform([1, len(p)], seed=seed)
        samples = tf.random.categorical(tf.math.log(logits), tf.reduce_prod(shape) if len(shape) > 0 else 1)
        return tf.reshape(tf.gather(tf.convert_to_tensor(p, dtype=dtype), samples[0]), shape)
    else:
        if hasattr(p, '__getitem__'):
            p = p[0]
        if tf.is_tensor(shape):
            return tf.cast(tf.fill(shape, p), dtype=dtype)
        return tf.constant(p, shape=shape, dtype=dtype)

def coarse_map(p, shape, size_percent, seed=1337, soft=False):
    small_map_shape = tf.concat([shape[:1], tf.cast(tf.cast([size_percent] * 2, tf.float32) * 100, tf.int32), shape[-1:]], axis=0)
    small_map = tf.cast(tf.random.uniform(small_map_shape, dtype=tf.float32, seed=seed) < p, tf.uint8)
    if soft == True:
        return tf.clip_by_value(tf.cast(tf.image.resize_bicubic(small_map, shape[1:3]), tf.float32), 0., 1.)
    return tf.cast(tf.image.resize(small_map, shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), tf.bool)
