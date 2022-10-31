"""
Loader for the material tagging class.
"""
import torch
import webdataset as wds

from compat2D import CompatLoader2D, COMPAT_ID


class MaterialTagLoader(CompatLoader2D):
    """
    Dataloader for the 2D material tagging task.
    Iterating over 2D renderings of shapes with:
        - part material labels
    """

    def __init__(self, root_url, split, n_comp, mat_list,
                 cache_dir=None, view_type=-1, transform=COMPAT_ID):
        super().__init__(root_url, split, n_comp,
                         cache_dir, view_type, transform)

        self.mat_index = {m:mat_list.index(m) for m in mat_list}
        self.mat_n = len(mat_list)

    def make_loader(self, batch_size, num_workers):
        # Transforming material lists to label tensor
        def to_tensor(parts_dict):
            label = torch.zeros(self.mat_n)
            for m in parts_dict.values():
              label[self.mat_index[m]] = 1
            return label

        # Instantiating dataset
        dataset = (
            super().make_loader()
            .decode("torchrgb")
            .to_tuple("render.png", "part_materials.json")
            .map_tuple(self.transform, to_tensor)
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
