PsyMo: Psychological traits from Motion

Each individual is coded as an integer ID ranging from 0 to 311.
In general, for particular walk, files have the naming convention "<ID>:<viewpoint>:<direction>:<run_id>:<variation>".

For example, "001:90:0:1:nm" refers to a walk from ID = 0, 90 degrees angle, direction 0, run 1 and normal walking variation.

The dataset is organized in several files, described below.

`walks.csv`
This .csv contains ID, viewpoint, variation, direction, run, file_id.
This file contains information about all captured walks, for each identity.

`metadata_raw_scores.csv`
This file contains the raw questionnaire scores for each individual.
The header has the naning convention of <QUESTIONNAIRE_NAME>_<SUBSCALE>.
Body information attributes about participants are also available (ATTR_* columns)

`metadata_labels.csv`
This file is similar to `metadata_raw_scores.csv`, but contains processed questionnaire responses, that should be used for classification of psychological traits. For each questionnaire, ordinal classes are computed based on known thresholds.

`semantic_data/smpl/<ID>/*.npz`
The .npz files contained here are SMPL body meshes, directly extracted from CLIFF (https://github.com/huawei-noah/noah-research/tree/master/CLIFF), and have the same internal structure:
    'imgname':     # image name, e.g., images/015601864.jpg, train2014/COCO_train2014_000000044474.jpg
    'center':      # bbox center, [x, y]
    'scale':       # bbox scale, bbox_size_in_pixel / 200.
    'part':        # 2D keypoint annotation, shape (24, 3), [x, y, conf], see common/skeleton_drawer.py for the order
    'annot_id':    # annotation ID, only available for the COCO dataset
    'pose':        # SMPL pose parameters in axis-angle, shape (72,)
    'shape':       # SMPL shape parameters, shape (10,)
    'has_smpl':    # whether the smpl parameters are available (true for all samples)
    'global_t':    # Pelvis translation in the camera coordinate system w.r.t the original full-frame image
    'focal_l':     # estimated focal length for the original image, np.sqrt(img_w ** 2 + img_h ** 2)
    'S':           # 3D joints with Pelvis aligned at (0, 0, 0), shape (24, 4), [x, y, z, conf], same order as 'part'

Files have the naming convention explained above.

`semantic_data/skeletons/<ID>/*.json`
These files contain 2D skeletons extracted from AlphaPose (https://github.com/MVIG-SJTU/AlphaPose). 2D keypoints are found in the "keypoints" key in the json.

`semantic_data/silhouettes/<ID>/**/*.png`
Extracted silhouettes using HTC (https://github.com/open-mmlab/mmdetection). Raw silhouettes are presented for each moving frame.

`semantic_data/silhouettes/<ID>/*_gei.png`
Gait Energy Images, for each sequence of silhouettes. These files are provided for convenience.