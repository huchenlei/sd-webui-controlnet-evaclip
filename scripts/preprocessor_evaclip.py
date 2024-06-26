import torch
import numpy as np
from typing import NamedTuple
from pathlib import Path
from einops import rearrange
from torchvision.transforms.functional import normalize, resize
from PIL import Image

from eva_clip.factory import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

# sd-webui-controlnet
from internal_controlnet.external_code import Preprocessor, PreprocessorParameter


class EvaCLIPResult(NamedTuple):
    id_cond_vit: torch.Tensor
    id_vit_hidden: torch.Tensor


class PreprocessorEvaCLIP(Preprocessor):
    def __init__(self):
        super().__init__(name="EVA02-CLIP-L-14-336")
        self.tags = []
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.returns_image = False
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 100  # higher goes to top in the list
        self.model = None
        self.eva_transform_mean = None
        self.eva_transform_std = None

    def load_model(self):
        """The model is around 800MB."""
        if self.model is None:
            self.model, _, _ = create_model_and_transforms(
                "EVA02-CLIP-L-14-336",
                "eva_clip",
                force_custom_clip=True,
                cache_dir=Path(__file__).parents[1] / "models",
                device=self.device,
            )
            eva_transform_mean = getattr(self.model.visual, "image_mean", OPENAI_DATASET_MEAN)
            eva_transform_std = getattr(self.model.visual, "image_std", OPENAI_DATASET_STD)
            if not isinstance(eva_transform_mean, (list, tuple)):
                eva_transform_mean = (eva_transform_mean,) * 3
            if not isinstance(eva_transform_std, (list, tuple)):
                eva_transform_std = (eva_transform_std,) * 3
            self.eva_transform_mean = eva_transform_mean
            self.eva_transform_std = eva_transform_std
        return self.model.to(device=self.device)

    def unload_model(self):
        """Unload to meta as CLIP model is large."""
        self.model.to(device="meta")

    def __call__(
        self,
        input_image,
        resolution=512,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs,
    ):
        self.load_model()
        if isinstance(input_image, np.ndarray):
            torch_img = torch.from_numpy(input_image).float() / 255.0
            input_image = rearrange(torch_img, "h w c -> 1 c h w")
        assert isinstance(input_image, torch.Tensor)
        assert input_image.ndim == 4, f"BCHW, {input_image.shape}"
        input_image = input_image.to(self.device)

        image_size = self.model.visual.image_size
        if isinstance(image_size, tuple):
            resize_target = image_size
        else:
            assert isinstance(image_size, int)
            resize_target = (image_size, image_size)

        face_features_image = resize(
            input_image,
            resize_target,
            Image.BICUBIC,
        )
        face_features_image = normalize(
            face_features_image, self.eva_transform_mean, self.eva_transform_std
        )
        id_cond_vit, id_vit_hidden = self.model.visual(
            face_features_image,
            return_all_features=False,
            return_hidden=True,
            shuffle=False,
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)
        return EvaCLIPResult(id_cond_vit, id_vit_hidden)


Preprocessor.add_supported_preprocessor(PreprocessorEvaCLIP())
