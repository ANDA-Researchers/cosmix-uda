import os
import torch
import yaml
import numpy as np
import tqdm

import MinkowskiEngine as ME
from utils.datasets.dataset import BaseDataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class NuScenesDataset(BaseDataset):
    def __init__(
        self,
        version: str = "full",
        phase: str = "train",
        dataset_path: str = "/data/nuscenes",
        mapping_path: str = "_resources/nuscenes_ns2sk.yaml",
        weights_path: str = None,
        voxel_size: float = 0.05,
        use_intensity: bool = False,
        augment_data: bool = False,
        sub_num: int = 50000,
        device: str = None,
        num_classes: int = 7,
        ignore_label: int = None,
    ):

        if weights_path is not None:
            weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
        super().__init__(
            version=version,
            phase=phase,
            dataset_path=dataset_path,
            voxel_size=voxel_size,
            sub_num=sub_num,
            use_intensity=use_intensity,
            augment_data=augment_data,
            device=device,
            num_classes=num_classes,
            ignore_label=ignore_label,
            weights_path=weights_path,
        )

        with open(mapping_path, "r") as stream:
            nuscenesyaml = yaml.safe_load(stream)

        if self.version == "full":
            self.split = {
                "train": nuscenesyaml["split"]["train"],
                "validation": nuscenesyaml["split"]["validation"],
            }
        elif self.version == "mini":
            self.split = {
                "train": nuscenesyaml["split"]["mini_train"],
                "validation": nuscenesyaml["split"]["mini_val"],
            }
        elif self.version == "sequential":
            self.split = {
                "train": nuscenesyaml["split"]["train"],
                "validation": nuscenesyaml["split"]["validation"],
            }
        else:
            raise NotImplementedError

        self.name = "nuScenesDataset"
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), "r"))

        self.pcd_path = []
        self.label_path = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        for sequence in self.split[self.phase]:
            num_frames = len(
                os.listdir(os.path.join(self.dataset_path, sequence, "labels"))
            )

            for f in np.arange(num_frames):
                pcd_path = os.path.join(
                    self.dataset_path,
                    "sequences",
                    f"{sequence:04d}",
                    "velodyne",
                    f"{int(f):06d}.bin",
                )
                label_path = os.path.join(
                    self.dataset_path,
                    "sequences",
                    f"{sequence:04d}",
                    "labels",
                    f"{int(f):06d}.label",
                )
                self.pcd_path.append(pcd_path)
                self.label_path.append(label_path)

        self.color_map = np.array(nuscenesyaml["color_map"]) / 255.0

    def __len__(self):
        return len(self.pcd_path)

    def __getitem__(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = self.load_label_kitti(label_tmp)
        points = pcd[:, :3]

        if self.use_intensity:
            colors = points[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {"points": points, "colors": colors, "labels": label}

        points = data["points"]
        colors = data["colors"]
        labels = data["labels"]

        sampled_idx = np.arange(points.shape[0])
        if self.phase == "train" and self.augment_data:
            sampled_idx = self.random_sample(points)
            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (points, np.ones((points.shape[0], 1), dtype=points.dtype))
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(
            points,
            colors,
            labels=labels,
            ignore_label=vox_ign_label,
            quantization_size=self.voxel_size,
            return_index=True,
        )

        missing_pts = self.sub_num - quantized_coords.shape[0]
        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None

        return {
            "coordinates": quantized_coords,
            "features": feats,
            "labels": labels,
            "sampled_idx": sampled_idx,
            "idx": torch.tensor(i),
        }

    def load_label_kitti(self, label_path: str):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert (sem_label + (inst_label << 16) == label).all()
        sem_label = self.remap_lut_val[sem_label]
        return sem_label.astype(np.int32)

    def get_dataset_weights(self):
        weights = np.zeros(self.remap_lut_val.max() + 1)
        for l in tqdm.tqdm(
            range(len(self.label_path)), desc="Loading weights", leave=True
        ):
            label_tmp = self.label_path[l]
            label = self.load_label_kitti(label_tmp)
            lbl, count = np.unique(label, return_counts=True)
            if self.ignore_label is not None:
                if self.ignore_label in lbl:
                    count = count[lbl != self.ignore_label]
                    lbl = lbl[lbl != self.ignore_label]

            weights[lbl] += count

        return weights

    def get_data(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = self.load_label_kitti(label_tmp)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = points[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {"points": points, "colors": colors, "labels": label}

        points = data["points"]
        colors = data["colors"]
        labels = data["labels"]

        return {"coordinates": points, "features": colors, "labels": labels, "idx": i}
