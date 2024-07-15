import h5py


class H5Record(h5py.File):
    def groups(self, subgroup=None):
        group = self[subgroup] if subgroup else self
        group_list = {}

        def _groups(name, obj):
            if isinstance(obj, h5py.Group):
                group_list[name] = obj

        group.visititems(_groups)
        return group_list

    def group_names(self, subgroup=None):
        group = self[subgroup] if subgroup else self
        group_list = []

        def _groups(name, obj):
            if isinstance(obj, h5py.Group):
                group_list.append(name)

        group.visititems(_groups)
        return group_list

    def datasets(self, subgroup=None):
        group = self[subgroup] if subgroup else self
        dset_list = {}

        def _dsets(name, obj):
            if isinstance(obj, h5py.Dataset):
                dset_list[name] = obj

        group.visititems(_dsets)
        return dset_list

    def dataset_names(self, subgroup=None):
        group = self[subgroup] if subgroup else self
        dset_list = []

        def _dsets(name, obj):
            if isinstance(obj, h5py.Dataset):
                dset_list.append(name)

        group.visititems(_dsets)
        return dset_list

    def count_datasets(self, subgroup=None):
        return len(self.dataset_names(subgroup))
