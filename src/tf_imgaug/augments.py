import tensorflow as tf
import math
import random

class AbstractAugment:

    def __init__(self, seed=1337):
        self.seed = seed

    def _set_seed(self, seed):
        self.seed = seed

    def _augment_images(self, images):
        return images
    def _augment_keypoints(self, keypoints):
        return keypoints
    def _augment_bboxes(self, bboxes):
        return bboxes

    def _init_rng(self):
        pass

    def __call__(self, images, keypoints=None, bboxes=None):
        res = list()
        self.last_shape = tf.shape(images)
        self._init_rng()

        res.append(self._augment_images(images))

        if not keypoints is None:
            res.append(self._augment_keypoints(keypoints))
        if not bboxes is None:
            res.append(self._augment_bboxes(bboxes))

        del self.last_shape

        return tuple(res)

class Noop(AbstractAugment):

    def _augment_images(self, images):
        return images

    def _augment_keypoints(self, keypoints):
        return keypoints

    def _augment_bboxes(self, bboxes):
        return bboxes

class Translate(AbstractAugment):

    def __init__(self, translate_percent, interpolation='BILINEAR'):
        super(Translate, self).__init__()
        self.translate_percent = translate_percent
        self.interpolation = interpolation

    def _init_rng(self):
        translations = tf.random_uniform(tf.concat([self.last_shape[:1], [2]], axis=0), seed=self.seed)
        translations = translations * [[
            self.translate_percent['x'][1] - self.translate_percent['x'][0],
            self.translate_percent['y'][1] - self.translate_percent['y'][0]
        ]] + [[self.translate_percent['x'][0] + self.translate_percent['y'][0]]]
        self.translations = translations * tf.expand_dims(tf.cast(self.last_shape[1:3], tf.float32), axis=0)

    def _augment_images(self, images):
        return tf.contrib.image.translate(images, self.translations, self.interpolation)

    def _augment_keypoints(self, keypoints):
        return keypoints + tf.expand_dims(self.translations, axis=1)

    def _augment_bboxes(self, bboxes):
        return bboxes + tf.expand_dims(tf.tile(self.translations, [1, 2]), axis=1)


class Rotate(AbstractAugment):

    def __init__(self, rotations, interpolation='BILINEAR'):
        super(Rotate, self).__init__()
        self.rotations = rotations
        self.interpolation = interpolation

    def _init_rng(self):
        angles = tf.random_uniform(self.last_shape[:1], seed=self.seed)
        self.angles = (angles * (self.rotations[1] - self.rotations[0]) + self.rotations[0]) * math.pi / 180

    def _augment_images(self, images):
        return tf.contrib.image.rotate(images, self.angles, self.interpolation)

    def _augment_keypoints(self, keypoints):
        angles = self.angles
        angles = tf.expand_dims(-angles, axis=1)
        shape = tf.cast(self.last_shape[1:3], tf.float32)

        keypoints = keypoints - (shape / 2)
        keypoints = tf.stack([
            tf.cos(angles) * keypoints[..., 0] - tf.sin(angles) * keypoints[..., 1],
            tf.sin(angles) * keypoints[..., 0] + tf.cos(angles) * keypoints[..., 1]
        ], axis=-1) + (shape / 2)


        return keypoints

    def _augment_bboxes(self, bboxes):
        angles = self.angles
        shape = tf.cast(self.last_shape[1:3], tf.float32)
        a = tf.cast((bboxes[..., 2] - bboxes[..., 0]) / 2, tf.float32)
        b = tf.cast((bboxes[..., 3] - bboxes[..., 1]) / 2, tf.float32)
        c_x = tf.cast((bboxes[..., 2] + bboxes[..., 0]) / 2, tf.float32)
        c_y = tf.cast((bboxes[..., 3] + bboxes[..., 1]) / 2, tf.float32)
        angles = tf.expand_dims(-angles, axis=1)

        angles = tf.cast(angles, tf.float32)
        a,b = tf.sqrt(tf.square(a * tf.cos(angles)) + tf.square(b * tf.sin(angles))), tf.sqrt(tf.square(a * tf.sin(angles)) + tf.square(b * tf.cos(angles)))

        c_x, c_y = (
            tf.cos(angles) * (c_x - shape[1] / 2) - tf.sin(angles) * (c_y - shape[0] / 2) + shape[1] / 2,
            tf.sin(angles) * (c_x - shape[1] / 2) + tf.cos(angles) * (c_y - shape[0] / 2) + shape[0] / 2
        )

        bboxes = tf.stack([
            c_x - a,
            c_y - b,
            c_x + a,
            c_y + b,
        ], axis=-1)
        return bboxes

class CropAndPad(AbstractAugment):

    def __init__(self, percent, pad_cval=0, mode='CONSTANT'):
        super(CropAndPad, self).__init__()
        self.percent = percent
        self.pad_cval = pad_cval
        self.mode = mode

    def _init_rng(self):
        crop_and_pads = tf.random_uniform((4,), seed=self.seed)
        crop_and_pads = crop_and_pads * [self.percent[1] - self.percent[0]] + self.percent[0]
        crop_and_pads = crop_and_pads * tf.cast(tf.concat([self.last_shape[1:3]] * 2, axis=0), tf.float32)
        self.crop_and_pads = tf.cast(crop_and_pads, tf.int32)

    def _augment_images(self, images):
        crop_and_pads = self.crop_and_pads

        crops = tf.clip_by_value(crop_and_pads, tf.minimum(0, tf.reduce_min(crop_and_pads)), 0)
        images = tf.slice(
            images,
            tf.concat([[0], -crops[:2], [0]], axis=0),
            #tf.concat([[-1], shape[1:3] + crops[2:], [-1]], axis=0)
            tf.concat([[-1], self.last_shape[1:3] + crops[:2] + crops[2:], [-1]], axis=0)
        )

        pads = tf.clip_by_value(crop_and_pads, 0, tf.maximum(0, tf.reduce_max(crop_and_pads)))
        pad_cval = tf.random_uniform((), minval=self.pad_cval[0], maxval=self.pad_cval[1], seed=self.seed) if type(self.pad_cval) == tuple else self.pad_cval
        images = tf.pad(images, tf.stack([[0, 0], pads[::2], pads[1::2], [0, 0]], axis=0), mode=self.mode, constant_values=pad_cval)

        return tf.image.resize_images(images, self.last_shape[1:3])

    def _augment_keypoints(self, keypoints):
        crop_and_pads = tf.cast(self.crop_and_pads, tf.float32)
        _shape = tf.cast(self.last_shape[1:3], tf.float32)
        shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))[..., ::-1]
        scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))[..., ::-1]
        keypoints = (keypoints + shift) / scale

        return keypoints

    def _augment_bboxes(self, bboxes):
        crop_and_pads = tf.cast(self.crop_and_pads, tf.float32)
        _shape = tf.cast(self.last_shape[1:3], tf.float32)
        shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))[..., ::-1]
        scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))[..., ::-1]
        bboxes = (bboxes + tf.tile(shift, (1, 1, 2))) / tf.tile(scale, (1, 1, 2))

        return bboxes

class Fliplr(AbstractAugment):

    def __init__(self, p=0.5):
        super(Fliplr, self).__init__()
        self.p = p

    def _init_rng(self):
        self.flip = tf.random_uniform((), seed=self.seed) < self.p

    def _augment_images(self, images):
        return tf.cond(
            self.flip,
            lambda: images[:, :, ::-1],
            lambda: images
        )

    def _augment_keypoints(self, keypoints):
        w = tf.cast(self.last_shape[1], tf.float32)
        return tf.cond(
            self.flip,
            lambda: tf.stack([w - keypoints[..., 0], keypoints[..., 1]], axis=-1),
            lambda: keypoints
        )

    def _augment_bboxes(self, bboxes):
        w = tf.cast(self.last_shape[1], tf.float32)
        return tf.cond(
            self.flip,
            lambda: tf.stack([w - bboxes[..., 2], bboxes[..., 1], w - bboxes[..., 0], bboxes[..., 3]], axis=-1),
            lambda: bboxes
        )

class Flipud(AbstractAugment):

    def __init__(self, p=0.5):
        super(Flipud, self).__init__()
        self.p = p

    def _init_rng(self):
        self.flip = tf.random_uniform((), seed=self.seed) < self.p

    def _augment_images(self, images):
        return tf.cond(
            self.flip,
            lambda: images[:, ::-1],
            lambda: images
        )

    def _augment_keypoints(self, keypoints):
        h = tf.cast(self.last_shape[2], tf.float32)
        return tf.cond(
            self.flip,
            lambda: tf.stack([keypoints[..., 0], h - keypoints[..., 1]], axis=-1),
            lambda: keypoints
        )

    def _augment_bboxes(self, bboxes):
        h = tf.cast(self.last_shape[2], tf.float32)
        return tf.cond(
            self.flip,
            lambda: tf.stack([bboxes[..., 0], h - bboxes[..., 3], bboxes[..., 2], h - bboxes[..., 1]], axis=-1),
            lambda: bboxes
        )


class Sometimes(AbstractAugment):

    def __init__(self, p, true_augment, false_augment=Noop()):
        super(Sometimes, self).__init__()
        self.p = p
        self.true_augment = true_augment
        self.false_augment = false_augment

        random.seed(self.seed)
        self.true_augment._set_seed(random.randint(0, 2 ** 64))
        self.false_augment._set_seed(random.randint(0, 2 ** 64))

    def _init_rng(self):
        self.flag = tf.random_uniform((), seed=self.seed) < self.p

    def __call__(self, *args, **kwargs):
        self._init_rng()
        return tf.cond(
            self.flag,
            lambda: self.true_augment(*args, **kwargs),
            lambda: self.false_augment(*args, **kwargs)
        )

class SomeOf(AbstractAugment):

    def __init__(self, num, children_augments):
        super(SomeOf, self).__init__()
        if type(num) == int:
            self.min_num = num
            self.max_num = num
        else:
            if num[0] is None:
                self.min_num = 0
            else:
                self.min_num = num[0]
            if num[1] is None:
                self.max_num = len(children_augments)
            else:
                self.max_num = num[1]
        self.children_augments = children_augments

    def _init_rng(self):
        self.probs = tf.random_uniform((len(self.children_augments),), seed=self.seed)
        self.count = tf.random_uniform((), minval=self.min_num, maxval=self.max_num + 1, dtype=tf.int32, seed=self.seed)

    def __call__(self, *args, **kwargs):
        self._init_rng()
        values, indices = tf.nn.top_k(self.probs, self.count)
        mask = tf.greater_equal(self.probs, tf.reduce_min(values))

        random.seed(self.seed)
        result = Noop()(*args, **kwargs)
        for i, augment in enumerate(self.children_augments):
            augment._set_seed(random.randint(0, 2 ** 64))
            result = tf.cond(
                mask[i],
                lambda: augment(*result),
                lambda: result
            )
        return result

class OneOf(SomeOf):

    def __init__(self, children_augments):
        super(OneOf, self).__init__((1, 1), children_augments)
