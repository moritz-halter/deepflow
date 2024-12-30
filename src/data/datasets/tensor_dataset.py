from torch.utils.data.dataset import Dataset


class TensorDataset(Dataset):
    def __init__(self, x, y, coords, chi=None, geo_feature=None):
        assert (x.size(0) == y.size(0)), "Size mismatch between tensors"
        if chi is not None:
            assert (x.size(0) == chi.size(0)), "Size mismatch between tensors"
        if geo_feature is not None:
            assert (x.size(0) == geo_feature.size(0)), "Size mismatch between tensors"

        self.x = x
        self.y = y
        self.chi = chi
        self.geo_feature = geo_feature
        self.coords = coords

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        coords = self.coords[index]

        res = {'x': x, 'y': y, 'coords': coords}
        if self.chi is not None:
            res['chi'] = self.chi[index]
        if self.geo_feature is not None:
            res['geo_feature'] = self.geo_feature[index]

        return res

    def __len__(self):
        return self.x.size(0)
