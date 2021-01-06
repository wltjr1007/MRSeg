# liver_spleen_segmentation
Liver and spleen segmentation using modified version of 2.5D DeepLabV3+


# Train from scracth
1. Modify `root_path`, `project_path`, `raw_data_path`, `result_path` from config.py.
1. Place a shape file at `raw_data_path`/segment/trn_shp.npy. This .npy file should contain size of z-axis of each image since the input data will be flattened.
1. Place a data file at `raw_data_path`/segment/trn_dat.npy. This .npy file should contain raw dicom pixel values in the shape of (N, 512,512).
1. Place a segmentation mask file at `raw_data_path`/segment/trn_lbl.npy. This .npy file should contain integer values in the shape of (N, 512,512), (0-background, 1-spleen, 2-liver).
1. Place a segmentation mask file at `raw_data_path`/trn_bound_mask_2d.npy. This .npy file should contain boolean values of the liver and spleen boundary in the shape of (N, 512, 512).
1. run `python main.py`.

# Run pretrained model
1. Download and unzip [pretrained model](https://github.com/wltjr1007/liver_spleen_segmentation/releases/download/v0/segement_model.zip).
1. Place .dcm files in a folder.
1. Modify `ckpt` and `in_dir` in test_seg.py.
1. run `python test_seg.py`
