import random

class Sequential:
    def __init__(self, augments, seed=random.randint(0, 2 ** 64)):
        self.augments = augments
        self.seed = seed

    def __call__(self, images, keypoints=None, bboxes=None):
        random.seed(self.seed)
        res = (images, keypoints, bboxes)
        for aug in self.augments:
            aug._set_seed(random.randint(0, 2 ** 64))
            res = aug(*res)

        return res

