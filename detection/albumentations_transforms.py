
#from albumentations import BaseCompose
from albumentations.core.composition import BaseCompose
class IfThen(BaseCompose):
    def __init__(self, ifpred, transform):
        super(IfThen, self).__init__([transform], p=1.0)
        self.ifpred = ifpred
        self.transform = transform

    def __call__(self, force_apply=False, **data):
        if self.replay_mode:
            for t in self.transform:
                data = t(**data)
            return data

        yes = self.ifpred(**data)
        if yes:
            data = self.transform(force_apply=force_apply, **data)
        return data

class IfThenElse(BaseCompose):
    def __init__(self, ifpred, transform, else_transform):
        super(IfThenElse, self).__init__([transform, else_transform], p=1.0)
        self.ifpred = ifpred
        self.transform = transform
        self.else_transform = else_transform

    def __call__(self, force_apply=False, **data):
        #if self.replay_mode:
        #    for t in self.transform:
        #        data = t(**data)
        #    return data

        yes = self.ifpred(**data)
        if yes:
            data = self.transform(force_apply=force_apply, **data)
        else:
            data = self.else_transform(force_apply=force_apply, **data)

        return data
