import torch
import numpy as np
from typing import NamedTuple
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from eva_clip.factory import create_model_and_transforms

# sd-webui-controlnet
from internal_controlnet.external_code import Preprocessor, PreprocessorParameter

# A1111
from modules import devices


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class EvaCLIPResult(NamedTuple):
    id_cond_vit: torch.Tensor
    id_vit_hidden: torch.Tensor


class PreprocessorEvaCLIP(Preprocessor):
    def __init__(self, device=None):
        super().__init__(name="EVA02-CLIP-L-14-336")
        self.tags = []
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 100  # higher goes to top in the list
        self.model = None
        self.eva_transform_mean = None
        self.eva_transform_std = None
        self.device = (
            devices.get_device_for("controlnet")
            if device is None
            else torch.device("cpu")
        )

    def load_model(self):
        """The model is around 800MB."""
        if self.model is None:
            model, _, _ = create_model_and_transforms(
                "EVA02-CLIP-L-14-336", "eva_clip", force_custom_clip=True
            )
            self.model = model.visual
            eva_transform_mean = getattr(
                self.clip_vision_model, "image_mean", OPENAI_DATASET_MEAN
            )
            eva_transform_std = getattr(
                self.clip_vision_model, "image_std", OPENAI_DATASET_STD
            )
            if not isinstance(eva_transform_mean, (list, tuple)):
                eva_transform_mean = (eva_transform_mean,) * 3
            if not isinstance(eva_transform_std, (list, tuple)):
                eva_transform_std = (eva_transform_std,) * 3
            self.eva_transform_mean = eva_transform_mean
            self.eva_transform_std = eva_transform_std
        return self.model.to(device=self.device)

    def unload_model(self):
        self.model.to(device="cpu")

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        self.load_model()
        if isinstance(input_image, np.ndarray):
            input_image = torch.from_numpy(input_image)
        assert isinstance(input_image, torch.Tensor)

        face_features_image = resize(
            input_image,
            self.clip_vision_model.image_size,
            InterpolationMode.BICUBIC,
        )
        face_features_image = normalize(
            face_features_image, self.eva_transform_mean, self.eva_transform_std
        )
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image,
            return_all_features=False,
            return_hidden=True,
            shuffle=False,
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)
        return EvaCLIPResult(id_cond_vit, id_vit_hidden)


Preprocessor.add_supported_preprocessor(PreprocessorEvaCLIP())
