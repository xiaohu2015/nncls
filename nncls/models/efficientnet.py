import copy
import math
from functools import partial
from typing import Any, Callable, Optional, List, Sequence

import torch
from torch import nn, Tensor

from .layers import ConvNormActivation, SqueezeExcitation, StochasticDepth
from ..utils import _log_api_usage_once
from .utils import _make_divisible


 __all__ = [
    "EfficientNet",
    "EfficientNet_B0_Weights",
    "EfficientNet_B1_Weights",
    "EfficientNet_B2_Weights",
    "EfficientNet_B3_Weights",
    "EfficientNet_B4_Weights",
    "EfficientNet_B5_Weights",
    "EfficientNet_B6_Weights",
    "EfficientNet_B7_Weights",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "expand_ratio={expand_ratio}"
        s += ", kernel={kernel}"
        s += ", stride={stride}"
        s += ", input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class
        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()
        _log_api_usage_once("models", self.__class__.__name__)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.extract_features(x))


def _efficientnet(
    width_mult: float,
    depth_mult: float,
    dropout: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> EfficientNet:

    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]

    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BICUBIC,
    "num_classes": 1000,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet",
}


class EfficientNet_B0_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=256, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (224, 224),
            "acc@1": 77.692,
            "acc@5": 93.532,
        },
    )
    default = ImageNet1K_V1


class EfficientNet_B1_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
        transforms=partial(ImageNetEval, crop_size=240, resize_size=256, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (240, 240),
            "acc@1": 78.642,
            "acc@5": 94.186,
        },
    )
    default = ImageNet1K_V1


class EfficientNet_B2_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
        transforms=partial(ImageNetEval, crop_size=288, resize_size=288, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (288, 288),
            "acc@1": 80.608,
            "acc@5": 95.310,
        },
    )
    default = ImageNet1K_V1


class EfficientNet_B3_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
        transforms=partial(ImageNetEval, crop_size=300, resize_size=320, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (300, 300),
            "acc@1": 82.008,
            "acc@5": 96.054,
        },
    )
    default = ImageNet1K_V1


class EfficientNet_B4_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
        transforms=partial(ImageNetEval, crop_size=380, resize_size=384, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (380, 380),
            "acc@1": 83.384,
            "acc@5": 96.594,
        },
    )
    default = ImageNet1K_V1


class EfficientNet_B5_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
        transforms=partial(ImageNetEval, crop_size=456, resize_size=456, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (456, 456),
            "acc@1": 83.444,
            "acc@5": 96.628,
        },
    )
    default = ImageNet1K_V1


class EfficientNet_B6_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
        transforms=partial(ImageNetEval, crop_size=528, resize_size=528, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (528, 528),
            "acc@1": 84.008,
            "acc@5": 96.916,
        },
    )
    default = ImageNet1K_V1


class EfficientNet_B7_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
        transforms=partial(ImageNetEval, crop_size=600, resize_size=600, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (600, 600),
            "acc@1": 84.122,
            "acc@5": 96.908,
        },
    )
    default = ImageNet1K_V1


@handle_legacy_interface(weights=("pretrained", EfficientNet_B0_Weights.ImageNet1K_V1))
def efficientnet_b0(
    *, weights: Optional[EfficientNet_B0_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B0_Weights.verify(weights)

    return _efficientnet(width_mult=1.0, depth_mult=1.0, dropout=0.2, weights=weights, progress=progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", EfficientNet_B1_Weights.ImageNet1K_V1))
def efficientnet_b1(
    *, weights: Optional[EfficientNet_B1_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B1_Weights.verify(weights)

    return _efficientnet(width_mult=1.0, depth_mult=1.1, dropout=0.2, weights=weights, progress=progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", EfficientNet_B2_Weights.ImageNet1K_V1))
def efficientnet_b2(
    *, weights: Optional[EfficientNet_B2_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B2_Weights.verify(weights)

    return _efficientnet(width_mult=1.1, depth_mult=1.2, dropout=0.3, weights=weights, progress=progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", EfficientNet_B3_Weights.ImageNet1K_V1))
def efficientnet_b3(
    *, weights: Optional[EfficientNet_B3_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B3_Weights.verify(weights)

    return _efficientnet(width_mult=1.2, depth_mult=1.4, dropout=0.3, weights=weights, progress=progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", EfficientNet_B4_Weights.ImageNet1K_V1))
def efficientnet_b4(
    *, weights: Optional[EfficientNet_B4_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B4_Weights.verify(weights)

    return _efficientnet(width_mult=1.4, depth_mult=1.8, dropout=0.4, weights=weights, progress=progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", EfficientNet_B5_Weights.ImageNet1K_V1))
def efficientnet_b5(
    *, weights: Optional[EfficientNet_B5_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B5_Weights.verify(weights)

    return _efficientnet(
        width_mult=1.6,
        depth_mult=2.2,
        dropout=0.4,
        weights=weights,
        progress=progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", EfficientNet_B6_Weights.ImageNet1K_V1))
def efficientnet_b6(
    *, weights: Optional[EfficientNet_B6_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B6_Weights.verify(weights)

    return _efficientnet(
        width_mult=1.8,
        depth_mult=2.6,
        dropout=0.5,
        weights=weights,
        progress=progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", EfficientNet_B7_Weights.ImageNet1K_V1))
def efficientnet_b7(
    *, weights: Optional[EfficientNet_B7_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B7_Weights.verify(weights)

    return _efficientnet(
        width_mult=2.0,
        depth_mult=3.1,
        dropout=0.5,
        weights=weights,
        progress=progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )
