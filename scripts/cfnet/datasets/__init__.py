from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .whu_dataset import WHUStereoDataset
from .igarss_dataset import IgarssDataset
from .custom_dataset import CustomDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "whu": WHUStereoDataset,
    "igarss": IgarssDataset,
    "custom": CustomDataset

}
