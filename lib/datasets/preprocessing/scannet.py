from pathlib import Path
from random import shuffle

import numpy as np
import sys
from lib.pc_utils import read_plyfile, save_point_cloud

SCANNET_RAW_PATH = Path('/media/ScanNet_data')
SCANNET_OUT_PATH = Path('/media/ScanNet_data/scans_pre')
TRAIN_DEST = 'train'
TEST_DEST = 'test'
SUBSETS = {TRAIN_DEST: 'scans', TEST_DEST: 'scans_test'}
POINTCLOUD_FILE = '_vh_clean_2.ply'
BUGS = {
    'train/scene0270_00_*.ply': 50,
    'train/scene0270_02_*.ply': 50,
    'train/scene0384_00_*.ply': 149,
}
# Preprocess data.
for out_path, in_path in SUBSETS.items():
    phase_out_path = SCANNET_OUT_PATH / out_path
    phase_out_path.mkdir(parents=True, exist_ok=True)
    for f in (SCANNET_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE):
        # Load pointcloud file.
        pointcloud = read_plyfile(f)
        # Make sure alpha value is meaningless.
        assert np.unique(pointcloud[:, -1]).size == 1
        # Load label file.
        label_f = f.parent / (f.stem + '.labels' + f.suffix)
        if label_f.is_file():
            label = read_plyfile(label_f)
            # Sanity check that the pointcloud and its label has same vertices.
            assert pointcloud.shape[0] == label.shape[0]
            assert np.allclose(pointcloud[:, :3], label[:, :3])
        else:  # Label may not exist in test case.
            label = np.zeros_like(pointcloud)
        xyz = pointcloud[:, :3]

        all_points = np.empty((0, 3))
        out_f = phase_out_path / (f.name[:-len(POINTCLOUD_FILE)] + f.suffix)
        processed = np.hstack((pointcloud[:, :6], np.array([label[:, -1]]).T))
        save_point_cloud(processed, out_f, with_label=True, verbose=False)

        # Check that all points are included in the crops.
        # assert set(tuple(l) for l in all_points.tolist()) == set(tuple(l) for l in xyz.tolist())
# Fix bug in the data.
for files, bug_index in BUGS.items():
    for f in SCANNET_OUT_PATH.glob(files):
        pointcloud = read_plyfile(f)
        bug_mask = pointcloud[:, -1] == bug_index
        print(f'Fixing {f} bugged label {bug_index} x {bug_mask.sum()}')
        pointcloud[bug_mask, -1] = 0
        save_point_cloud(pointcloud, f, with_label=True, verbose=False)
