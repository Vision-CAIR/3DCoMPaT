"""
Pre-configured wrappers for the 2D CoMPaT loaders based on the CoMPaT10 and CoMPaT50 tasks.
"""

from compat2D import ShapeLoader, SegmentationLoader, GCRLoader


""""
CoMPaT10 dataloaders

    Args:
        root_url:     Base dataset URL containing data split shards
        split:        One of {train, valid}.
        transform:    Transform to be applied on rendered views

        batch_size:   Size of each databatch
        num_workers:  Number of processes to use
"""

def CoMPaT10_ShapeLoader(root_url, split, transform=None, batch_size=64, num_workers=2):
    return ShapeLoader(root_url = root_url,
                       split    = split,
                       n_comp   = 10,
                       transform= transform) \
    .make_loader(batch_size=batch_size, num_workers=num_workers)

def CoMPaT10_SegmentationLoader(root_url, split, transform=None, batch_size=64, num_workers=2):
    return SegmentationLoader(root_url  = root_url,
                              split     = split,
                              n_comp    = 10,
                              transform = transform) \
    .make_loader(batch_size=batch_size, num_workers=num_workers)

def CoMPaT10_GCRLoader(root_url, split, transform=None, batch_size=64, num_workers=2):
    return GCRLoader(root_url  = root_url,
                     split     = split,
                     n_comp    = 10,
                     transform = transform) \
    .make_loader(batch_size=batch_size, num_workers=num_workers)


""""
CoMPaT50 dataloaders

    Args:
        root_url:     Base dataset URL containing data split shards
        split:        One of {train, valid}.
        transform:    Transform to be applied on rendered views

        batch_size:   Size of each databatch
        num_workers:  Number of processes to use
"""

def CoMPaT50_ShapeLoader(root_url, split, transform=None, batch_size=64, num_workers=2):
    return ShapeLoader(root_url = root_url,
                       split    = split,
                       n_comp   = 50,
                       transform= transform) \
    .make_loader(batch_size=batch_size, num_workers=num_workers)

def CoMPaT50_SegmentationLoader(root_url, split, transform=None, batch_size=64, num_workers=2):
    return SegmentationLoader(root_url  = root_url,
                              split     = split,
                              n_comp    = 50,
                              transform = transform) \
    .make_loader(batch_size=batch_size, num_workers=num_workers)

def CoMPaT50_GCRLoader(root_url, split, transform=None, batch_size=64, num_workers=2):
    return GCRLoader(root_url  = root_url,
                     split     = split,
                     n_comp    = 50,
                     transform = transform) \
    .make_loader(batch_size=batch_size, num_workers=num_workers)
