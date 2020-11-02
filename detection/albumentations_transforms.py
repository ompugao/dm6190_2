
#from albumentations import BaseCompose
from albumentations.core.composition import BaseCompose
class IfThen(BaseCompose):
    def __init__(self, ifpred, transform):
        super(IfThen, self).__init__(transforms, p)
        self.ifpred = ifpred
        self.transform = transform

    def __call__(self, force_apply=False, **data):
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        yes = self.ifpred(**data)
        if yes:
            data = self.transform(force_apply=force_apply, **data)
        return data
