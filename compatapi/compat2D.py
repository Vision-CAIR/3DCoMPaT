""""
Dataloaders for the 2D 3DCoMPaT tasks.
"""

import json
import os
import webdataset as wds


COMPAT_ID = lambda x:x


class CompatLoader2D:
    """
    Base class for 2D dataset loaders.

    Args:
        root_url:    Base dataset URL containing data split shards
        split:       One of {train, valid}.
        n_comp:      Number of compositions to use
        cache_dir:   Cache directory to use
        view_type:   Filter by view type [0: canonical views, 1: random views]
        transform:   Transform to be applied on rendered views
    """
    def __init__(self, root_url, split, n_comp, cache_dir=None, view_type=-1, transform=COMPAT_ID):
        if view_type not in [-1, 0, 1]:
            raise RuntimeError("Invalid argument: view_type can only be [-1, 0, 1]")
        if split not in ["train", "valid"]:
            raise RuntimeError("Invalid split: [%s]." % split)

        root_url = os.path.normpath(root_url)

        # Reading sample count from metadata
        datacount = json.load(open(root_url + "/datacount.json", "r"))
        sample_count = datacount['sample_count']
        max_comp     = datacount['compositions']

        if n_comp > max_comp:
            except_str = "Required number of compositions exceeds maximum available in [%s] (%d)." % (root_url, n_comp)
            raise RuntimeError(except_str)

        # Computing dataset size
        self.dataset_size = sample_count[split]*n_comp

        # Configuring size of shuffle buffer
        if split == "train":
            self.shuffle = 1000
        elif split == "valid":
            self.shuffle = 0

        # Formatting WebDataset base URL
        comp_str = "%04d" % (n_comp -1)
        self.url = '%s/%s/%s_{0000..%s}.tar' % (root_url, split, split, comp_str)


        self.cache_dir = cache_dir
        self.transform = transform
        self.view_type = view_type


    def make_loader(self):
        """
        Instantiating dataloader

        Args:
            batch_size:  Size of each batch in the loader
            num_workers: Number of process workers to use when loading data
        """
        # Instantiating dataset
        if self.cache_dir:
            dataset = wds.WebDataset(self.url)
        else:
            dataset = wds.WebDataset(self.url, cache_dir=self.cache_dir)

        if self.view_type != -1:
            view_val = bytes(str(self.view_type), 'utf-8')
            dataset  = dataset.select(lambda x: x["view_type.cls"] == view_val)
        dataset = dataset.shuffle(self.shuffle)

        return dataset


class ShapeLoader(CompatLoader2D):
    """
    Shape classification data loader.
    Iterating over 2D renderings of shapes with a shape category label.

    Args:
        -> CompatLoader2D
    """

    def __init__(self, root_url, split, n_comp, cache_dir=None, view_type=-1, transform=COMPAT_ID):
        super().__init__(root_url, split, n_comp, cache_dir, view_type, transform)


    def make_loader(self, batch_size, num_workers):
        # Instantiating dataset
        dataset = (
            super().make_loader()
            .decode("torchrgb")
            .to_tuple("render.png", "target.cls")
            .map_tuple(self.transform, COMPAT_ID)
            .batched(batch_size, partial=False)
        )

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader


class SegmentationLoader(CompatLoader2D):
    """
    Part segmentation and shape classification data loader.
    Iterating over 2D renderings of shapes with a shape category label, and a segmentation mask with pixel coding for parts.

    Args:
        -> CompatLoader2D

        mask_transform:  Transform to apply on segmentation masks
        code_transform:  Function to apply on segmentation mask labels 
    """

    def __init__(self, root_url, split, n_comp, cache_dir=None, view_type=-1,
                    transform=COMPAT_ID, mask_transform = COMPAT_ID,
                    code_transform = COMPAT_ID):
        super().__init__(root_url, split, n_comp, cache_dir, view_type, transform)

        self.mask_transform = mask_transform
        self.code_transform = code_transform


    def make_loader(self, batch_size, num_workers):
        # Instantiating dataset
        dataset = (
            super().make_loader()
            .decode("torchrgb")
            .to_tuple("render.png", "target.cls", "mask.png", "mask_code.json")
            .map_tuple(self.transform, COMPAT_ID, self.mask_transform, self.code_transform)
            .batched(batch_size, partial=False)
        )

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader


class GCRLoader(CompatLoader2D):
    """
    Dataloader for the full 2D compositional task.
    Iterating over 2D renderings of shapes with:
        - shape category label
        - segmentation mask with pixel coding for parts
        - part materials labels

    Args:
        -> CompatLoader2D

        mask_transform:      Transform to apply on segmentation masks
        code_transform:      Function to apply on segmentation mask labels 
        part_mat_transform:  Function to apply on part-material labels
    """

    def __init__(self, root_url, split, n_comp, cache_dir=None, view_type=-1,
                    transform=COMPAT_ID, mask_transform = COMPAT_ID,
                    code_transform = COMPAT_ID, part_mat_transform = COMPAT_ID):
        super().__init__(root_url, split, n_comp, cache_dir, view_type, transform)

        self.mask_transform = mask_transform
        self.code_transform = code_transform
        self.part_mat_transform = part_mat_transform


    def make_loader(self, batch_size, num_workers):
        # Instantiating dataset
        dataset = (
            super().make_loader()
            .decode("torchrgb")
            .to_tuple("render.png", "target.cls", "mask.png", "mask_code.json", "part_materials.json")
            .map_tuple(self.transform, COMPAT_ID, self.mask_transform, self.code_transform, self.part_mat_transform)
            .batched(batch_size, partial=False)
        )

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader
