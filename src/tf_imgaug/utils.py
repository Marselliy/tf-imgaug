import tensorflow as tf

def p_to_tensor(p, shape, dtype=tf.float32, seed=1337):
    if type(p) == tuple and p[0] != p[1]:
        return tf.random_uniform(shape=shape, dtype=dtype, minval=p[0], maxval=p[1], seed=seed)
    elif type(p) == list and len(p) != 1:
        logits = tf.random_uniform([1, len(p)], seed=seed)
        samples = tf.multinomial(tf.log(logits), tf.reduce_prod(shape))
        return tf.reshape(tf.gather(tf.convert_to_tensor(p, dtype=dtype), samples[0]), shape)
    else:
        if hasattr(p, '__getitem__'):
            p = p[0]
        if type(shape) == tf.Tensor:
            return tf.cast(tf.fill(shape, p), dtype=dtype)
        return tf.constant(p, shape=shape, dtype=dtype)