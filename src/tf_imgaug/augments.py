import tensorflow as tf
import math
import random

from utils import p_to_tensor, coarse_map

class AbstractAugment:

    def __init__(self, seed=1337, separable=True):
        self.seed = seed
        self.separable = separable

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

    def __call__(self, images, keypoints, bboxes):
        self.last_shape = tf.shape(images)

        def _aug(e):
            self._init_rng()
            return (
                self._augment_images(e[0]), 
                self._augment_keypoints(e[1]),
                self._augment_bboxes(e[2])
            )

        if self.separable:
            images_aug, keypoints_aug, bboxes_aug = tf.map_fn(_aug, tuple([tf.expand_dims(e, axis=1) for e in (images, keypoints, bboxes)]))
            return images_aug[:, 0], keypoints_aug[:, 0], bboxes_aug[:, 0]

        return _aug((images, keypoints, bboxes))

class Noop(AbstractAugment):

    def __init__(self):
        super(Noop, self).__init__(separable=False)

    def _augment_images(self, images):
        return images

    def _augment_keypoints(self, keypoints):
        return keypoints

    def _augment_bboxes(self, bboxes):
        return bboxes

class Translate(AbstractAugment):

    def __init__(self, translate_percent, interpolation='BILINEAR'):
        super(Translate, self).__init__(separable=False)
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
        super(Rotate, self).__init__(separable=False)
        self.rotations = rotations
        self.interpolation = interpolation

    def _init_rng(self):
        self.angles = p_to_tensor(self.rotations, shape=self.last_shape[:1], seed=self.seed) * math.pi / 180

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
        crop_and_pads = p_to_tensor(self.percent, shape=(4,), dtype=tf.float32)
        crop_and_pads = crop_and_pads * tf.cast(tf.concat([self.last_shape[1:3]] * 2, axis=0), tf.float32)
        self.crop_and_pads = tf.cast(crop_and_pads, tf.int32)

    def _augment_images(self, images):
        crop_and_pads = self.crop_and_pads

        crops = tf.clip_by_value(crop_and_pads, tf.minimum(0, tf.reduce_min(crop_and_pads)), 0)
        images = tf.slice(
            images,
            tf.concat([[0], -crops[:2], [0]], axis=0),
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
            lambda: tf.image.flip_left_right(images),
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
            lambda: tf.image.flip_up_down(images),
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
        self.true_augment._set_seed(random.randint(0, 2 ** 32))
        self.false_augment._set_seed(random.randint(0, 2 ** 32))
        self.true_augment.separable = False
        self.false_augment.separable = False

    def _init_rng(self):
        self.flag = tf.random_uniform((), seed=self.seed) < self.p

    def __call__(self, images, keypoints, bboxes):
        def _aug(e):
            self._init_rng()
            return tf.cond(
                self.flag,
                lambda: self.true_augment(*e),
                lambda: self.false_augment(*e)
            )

        if self.separable:
            images_aug, keypoints_aug, bboxes_aug = tf.map_fn(_aug, tuple([tf.expand_dims(e, axis=1) for e in (images, keypoints, bboxes)]))
            return images_aug[:, 0], keypoints_aug[:, 0], bboxes_aug[:, 0]
        else:
            return _aug((images, keypoints, bboxes))

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

        for aug in self.children_augments:
            aug.separable = False

    def _init_rng(self):
        self.probs = tf.random_uniform((len(self.children_augments),), seed=self.seed)
        self.count = tf.random_uniform((), minval=self.min_num, maxval=self.max_num + 1, dtype=tf.int32, seed=self.seed)

    def __call__(self, images, keypoints, bboxes):
        def _aug(e):
            self._init_rng()
            values, _ = tf.nn.top_k(self.probs, self.count)
            mask = tf.greater_equal(self.probs, tf.reduce_min(values))

            random.seed(self.seed)
            result = Noop()(*e)
            for i, augment in enumerate(self.children_augments):
                augment._set_seed(random.randint(0, 2 ** 32))
                result = tf.cond(
                    mask[i],
                    lambda: augment(*result),
                    lambda: result
                )
            return result

        if self.separable:
            images_aug, keypoints_aug, bboxes_aug = tf.map_fn(_aug, tuple([tf.expand_dims(e, axis=1) for e in (images, keypoints, bboxes)]))
            return images_aug[:, 0], keypoints_aug[:, 0], bboxes_aug[:, 0]
        else:
            return _aug((images, keypoints, bboxes))

class OneOf(SomeOf):

    def __init__(self, children_augments):
        super(OneOf, self).__init__((1, 1), children_augments)

class AbstractNoise(AbstractAugment):

    def __init__(self, noise_range, p, per_channel, coarse, size_percent=0.01, seed=1337, separable=False):
        super(AbstractNoise, self).__init__(seed=seed, separable=separable)

        self.noise_range = noise_range
        self.p = p
        self.per_channel = per_channel
        self.coarse = coarse
        self.size_percent = size_percent

    def _init_rng(self):
        if self.per_channel:
            noise_shape = self.last_shape
        else:
            noise_shape = tf.concat([self.last_shape[:-1], [1]], axis=0)

        p = p_to_tensor(self.p, tf.concat([noise_shape[:1], [1, 1], noise_shape[-1:]], axis=0), seed=self.seed)
        if self.coarse:
            size_percent = p_to_tensor(self.size_percent, (), seed=self.seed - 1)
            if self.per_channel:
                map_shape = self.last_shape
            else:
                map_shape = tf.concat([self.last_shape[:-1], [1]], axis=0)
            self.mask = coarse_map(p, map_shape, size_percent, seed=self.seed)
        else:
            self.mask = tf.random_uniform(shape=noise_shape, seed=self.seed) < p
            self.mask = tf.cast(self.mask, tf.float32)

        self.noise = p_to_tensor(self.noise_range, noise_shape, seed=self.seed)

    def _augment_images(self, images):
        if self.p == 0:
            return images
        if self.p == 1:
            return tf.broadcast_to(self.noise, self.last_shape)
        return images * (1 - self.mask) + self.noise * self.mask

class Salt(AbstractNoise):

    def __init__(self, p=0):
        super(Salt, self).__init__(noise_range=(128, 255), p=p, per_channel=False, coarse=False)

class CoarseSalt(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01):
        super(CoarseSalt, self).__init__(noise_range=(128, 255), p=p, per_channel=False, coarse=True, size_percent=size_percent)

class Pepper(AbstractNoise):

    def __init__(self, p=0):
        super(Pepper, self).__init__(noise_range=(0, 128), p=p, per_channel=False, coarse=False)

class CoarsePepper(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01):
        super(CoarsePepper, self).__init__(noise_range=(0, 128), p=p, per_channel=False, coarse=True, size_percent=size_percent)

class SaltAndPepper(AbstractNoise):

    def __init__(self, p=0):
        super(SaltAndPepper, self).__init__(noise_range=(0, 255), p=p, per_channel=False, coarse=False)

class CoarseSaltAndPepper(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01):
        super(CoarseSaltAndPepper, self).__init__(noise_range=(0, 255), p=p, per_channel=False, coarse=True, size_percent=size_percent)

class Dropout(AbstractNoise):

    def __init__(self, p=0, per_channel=False):
        super(Dropout, self).__init__(noise_range=0, p=p, per_channel=per_channel, coarse=False)

class CoarseDropout(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01, per_channel=False):
        super(CoarseDropout, self).__init__(noise_range=0, p=p, per_channel=per_channel, coarse=True, size_percent=size_percent)

class JpegCompression(AbstractAugment):

    def __init__(self, quality, seed=1337, separable=True):
        super(JpegCompression, self).__init__(seed=seed, separable=separable)
        self.quality = quality

    def _augment_images(self, images):
        if type(self.quality) == tuple:
            min_quality, max_quality = self.quality
        else:
            min_quality, max_quality = self.quality, self.quality + 1

        images = tf.cast(tf.clip_by_value(images, 0., 255.), tf.uint8)
        return tf.cast(tf.expand_dims(tf.image.random_jpeg_quality(images[0], min_quality, max_quality, seed=self.seed), axis=0), tf.float32)

class AdditiveGaussianNoise(AbstractAugment):

    def __init__(self, scale, per_channel=True, seed=1337):
        super(AdditiveGaussianNoise, self).__init__(seed=seed, separable=False)
        self.scale = scale
        self.per_channel = per_channel
    
    def _augment_images(self, images):
        scale = p_to_tensor(self.scale, tf.concat([self.last_shape[:1], [1, 1, 1]], axis=0), seed=self.seed) * 255
        if self.per_channel:
            noise_shape = self.last_shape
        else:
            noise_shape = tf.concat([self.last_shape[:-1], [1]], axis=0)
        return tf.clip_by_value(images + tf.random_normal(noise_shape) * scale, 0, 255)

class Grayscale(AbstractAugment):

    def __init__(self, p, seed=1337):
        super(Grayscale, self).__init__(seed=seed, separable=False)
        self.p = p
    
    def _augment_images(self, images):
        p = p_to_tensor(self.p, tf.concat([self.last_shape[:1], [1, 1, 1]], axis=0), seed=self.seed)
        rgb_weights = [0.2989, 0.5870, 0.1140]
        r = \
            images[..., :1] * (p * (rgb_weights[0] - 1) + 1) + \
            images[..., 1:2] * p * rgb_weights[1] + \
            images[..., 2:3] * p * rgb_weights[2]
        g = \
            images[..., :1] * p * rgb_weights[0] + \
            images[..., 1:2] * (p * (rgb_weights[1] - 1) + 1) + \
            images[..., 2:3] * p * rgb_weights[2]
        b = \
            images[..., :1] * p * rgb_weights[0] + \
            images[..., 1:2] * p * rgb_weights[1] + \
            images[..., 2:3] * (p * (rgb_weights[2] - 1) + 1)

        return tf.concat([r, g, b], axis=-1)

