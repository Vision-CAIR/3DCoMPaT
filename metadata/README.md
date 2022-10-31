# 3D CoMPaT: Metadata

We provide here information on the 3DCoMPaT metadata files in this folder.

### Model IDs

3DCoMPaT models are indexed by a unique identifier, which a 36 characters string of the form:

`deadb33f-dead-b33f-b33f-b33ff0000000`

This ID is used in the GLB files provided, in the WebDatase tar entries and in the 3D textured model files.

### Classes information

- `classes.json`: The ordered list of classes in 3DCoMPaT.

- `materials.json`: The ordered list of material categories in 3DCoMPaT.

- `parts.json`: The ordered list of parts in 3DCoMPaT.

### Models information

- `labels.json`: Maps each model ID to a class index. To get the class string, you can use the `classes.json` file.

- `part_material_map.json`: Maps each model ID to a dictionary of the form:

```json
{"model_id_0" : {"part_name_0": "material_category_0",
                 "part_name_1": "material_category_1",
                 ... },
  ...
}
```

These material categories are valid for all stylized models and rendered views provided in the dataset.

### Data splits

The train, test and validation splits are provided in the `split.csv` file.
