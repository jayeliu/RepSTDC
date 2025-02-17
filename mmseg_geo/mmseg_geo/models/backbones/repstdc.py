import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential
from mmpretrain.models.backbones.repvgg import RepVGGBlock
from mmseg.models.backbones.stdc import FeatureFusionModule
from mmseg.models.utils import resize
from mmengine.registry import MODELS
from mmseg.models.backbones.bisenetv1 import AttentionRefinementModule
from .CoordinateAttention import CoordAtt
from .CBAM import CBAMBlock
from .ECAAttention import ECAAttention
class CoordinateFeatureFusionModule(BaseModule):
    """Coordinate Feature Fusion Module. This module is different from FeatureFusionModule
    in BiSeNetV1. It uses two ConvModules in `self.attention` whose inter
    channel number is calculated by given `scale_factor`, while
    FeatureFusionModule in BiSeNetV1 only uses one ConvModule in
    `self.conv_atten`.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scale_factor (int): The number of channel scale factor.
            Default: 4.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): The activation config for conv layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=4,
        neck=False,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        init_cfg=None,
    ):
        super(CoordinateFeatureFusionModule, self).__init__(init_cfg=init_cfg)
        self.conv0 = ConvModule(in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.attention=CoordAtt(inp=out_channels,oup=out_channels,reduction=scale_factor)
       

    def forward(self, spatial_inputs, context_inputs):
        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(inputs)
        x_attn=self.attention(x)
        return x+x_attn


class CBAMFeatureFusionModule(BaseModule):
    """Coordinate Feature Fusion Module. This module is different from FeatureFusionModule
    in BiSeNetV1. It uses two ConvModules in `self.attention` whose inter
    channel number is calculated by given `scale_factor`, while
    FeatureFusionModule in BiSeNetV1 only uses one ConvModule in
    `self.conv_atten`.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scale_factor (int): The number of channel scale factor.
            Default: 4.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): The activation config for conv layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=4,
        neck=False,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        init_cfg=None,
    ):
        super(CBAMFeatureFusionModule, self).__init__(init_cfg=init_cfg)
        self.conv0 = ConvModule(in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.attention=CBAMBlock(channel=out_channels,reduction=scale_factor)
       

    def forward(self, spatial_inputs, context_inputs):
        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(inputs)
        x_attn=self.attention(x)
        return x_attn
    
class ECAAttentionRefinementModule(BaseModule):
    """Attention Refinement Module (ARM) to refine the features of each stage.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Attention Refinement Module.
    """

    def __init__(
        self,
        in_channels,
        out_channel,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        init_cfg=None,
    ):
        super(ECAAttentionRefinementModule, self).__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        self.eca_atten=ECAAttention()


    def forward(self, x):
        x = self.conv_layer(x)
        x_atten=self.eca_atten(x)
        x_out=x*x_atten
        return x_out

class CoordinateAttentionRefinementModule(BaseModule):
    """Attention Refinement Module (ARM) to refine the features of each stage.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Attention Refinement Module.
    """

    def __init__(
        self,
        in_channels,
        out_channel,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        init_cfg=None,
    ):
        super(CoordinateAttentionRefinementModule, self).__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        # self.conv_layer=RepVGGBlock(in_channels, out_channel, norm_cfg=norm_cfg)
        self.atten_conv_layer_h = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            ConvModule(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            nn.Sigmoid(),
        )
        self.atten_conv_layer_w = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            ConvModule(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x_atten_h = self.atten_conv_layer_h(x)
        x_atten_w = self.atten_conv_layer_w(x)
        x_out = (x * x_atten_h+x * x_atten_w)/2
        return x_out


# from ..builder import BACKBONES, build_backbone
# from .bisenetv1 import AttentionRefinementModule
class RepSTDCModule(BaseModule):
    """STDCModule.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels before scaling.
        stride (int): The number of stride for the first conv layer.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): The activation config for conv layers.
        num_convs (int): Numbers of conv layers.
        fusion_type (str): Type of fusion operation. Default: 'add'.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        norm_cfg=None,
        act_cfg=None,
        se_cfg=None,
        num_convs=4,
        fusion_type="add",
        with_cp=False,
        init_cfg=None,
        deploy=False,
    ):
        super(RepSTDCModule, self).__init__(init_cfg=init_cfg)
        assert num_convs > 1
        assert fusion_type in ["add", "cat"]
        self.stride = stride
        self.deploy = deploy
        self.with_downsample = True if self.stride == 2 else False
        self.fusion_type = fusion_type

        self.layers = ModuleList()
        # conv_0 = RepVGGBlock(
        #     in_channels,
        #     out_channels // 2,
        #     norm_cfg=norm_cfg,
        #     se_cfg=se_cfg,
        #     with_cp=with_cp,
        # )
        conv_0 = ConvModule(
            in_channels, out_channels // 2, kernel_size=1, norm_cfg=norm_cfg
        )

        if self.with_downsample:
            self.downsample = RepVGGBlock(
                out_channels // 2,
                out_channels // 2,
                stride=2,
                groups=out_channels // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                se_cfg=se_cfg,
                with_cp=with_cp,
            )
            # self.downsample = ConvModule(
            #     out_channels // 2,
            #     out_channels // 2,
            #     kernel_size=3,
            #     stride=2,
            #     padding=1,
            #     groups=out_channels // 2,
            #     norm_cfg=norm_cfg,
            #     act_cfg=None)

            if self.fusion_type == "add":
                self.layers.append(nn.Sequential(conv_0, self.downsample))
                self.skip = Sequential(
                    # ConvModule(
                    #     in_channels,
                    #     in_channels,
                    #     kernel_size=3,
                    #     stride=2,
                    #     padding=1,
                    #     groups=in_channels,
                    #     norm_cfg=norm_cfg,
                    #     act_cfg=None,
                    # ),
                    RepVGGBlock(
                        in_channels,
                        in_channels,
                        stride=2,
                        groups=in_channels,
                        norm_cfg=norm_cfg,
                        with_cp=with_cp,
                        se_cfg=se_cfg,
                    ),
                    ConvModule(
                        in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None
                    ),
                )
            else:
                self.layers.append(conv_0)
                self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.layers.append(conv_0)

        for i in range(1, num_convs):
            out_factor = 2 ** (i + 1) if i != num_convs - 1 else 2**i
            self.layers.append(
                # ConvModule(
                #     out_channels // 2**i,
                #     out_channels // out_factor,
                #     kernel_size=3,
                #     stride=1,
                #     padding=1,
                #     norm_cfg=norm_cfg,
                #     act_cfg=act_cfg,
                # )
                RepVGGBlock(
                    out_channels // 2**i,
                    out_channels // out_factor,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                    se_cfg=se_cfg,
                )
            )

    def forward(self, inputs):
        if self.fusion_type == "add":
            out = self.forward_add(inputs)
        else:
            out = self.forward_cat(inputs)
        return out

    def forward_add(self, inputs):
        layer_outputs = []
        x = inputs.clone()
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            inputs = self.skip(inputs)

        return torch.cat(layer_outputs, dim=1) + inputs

    def forward_cat(self, inputs):
        x0 = self.layers[0](inputs)
        layer_outputs = [x0]
        for i, layer in enumerate(self.layers[1:]):
            if i == 0:
                if self.with_downsample:
                    x = layer(self.downsample(x0))
                else:
                    x = layer(x0)
            else:
                x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            layer_outputs[0] = self.skip(x0)
        return torch.cat(layer_outputs, dim=1)

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        self.deploy = True


@MODELS.register_module()
class RepSTDCNet(BaseModule):
    """This backbone is the implementation of `Rethinking BiSeNet For Real-time
    Semantic Segmentation <https://arxiv.org/abs/2104.13188>`_.

    Args:
        stdc_type (int): The type of backbone structure,
            `STDCNet1` and`STDCNet2` denotes two main backbones in paper,
            whose FLOPs is 813M and 1446M, respectively.
        in_channels (int): The num of input_channels.
        channels (tuple[int]): The output channels for each stage.
        bottleneck_type (str): The type of STDC Module type, the value must
            be 'add' or 'cat'.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): The activation config for conv layers.
        num_convs (int): Numbers of conv layer at each STDC Module.
            Default: 4.
        with_final_conv (bool): Whether add a conv layer at the Module output.
            Default: True.
        pretrained (str, optional): Model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> import torch
        >>> stdc_type = 'STDCNet1'
        >>> in_channels = 3
        >>> channels = (32, 64, 256, 512, 1024)
        >>> bottleneck_type = 'cat'
        >>> inputs = torch.rand(1, 3, 1024, 2048)
        >>> self = STDCNet(stdc_type, in_channels,
        ...                 channels, bottleneck_type).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 256, 128, 256])
        outputs[1].shape = torch.Size([1, 512, 64, 128])
        outputs[2].shape = torch.Size([1, 1024, 32, 64])
    """

    arch_settings = {
        "STDCNet1": [(2, 1), (2, 1), (2, 1)],
        "STDCNet2": [(2, 1, 1, 1), (2, 1, 1, 1, 1), (2, 1, 1)],
    }

    def __init__(
        self,
        stdc_type,
        in_channels,
        channels,
        bottleneck_type,
        norm_cfg,
        act_cfg,
        num_convs=4,
        with_final_conv=False,
        pretrained=None,
        deploy=False,
        init_cfg=None,
    ):
        super(RepSTDCNet, self).__init__(init_cfg=init_cfg)
        assert (
            stdc_type in self.arch_settings
        ), f"invalid structure {stdc_type} for STDCNet."
        assert bottleneck_type in [
            "add",
            "cat",
        ], f"bottleneck_type must be `add` or `cat`, got {bottleneck_type}"

        assert (
            len(channels) == 5
        ), f"invalid channels length {len(channels)} for STDCNet."

        self.in_channels = in_channels
        self.channels = channels
        self.stage_strides = self.arch_settings[stdc_type]
        self.prtrained = pretrained
        self.num_convs = num_convs
        self.with_final_conv = with_final_conv
        self.deploy = deploy
        self.act_cfg = act_cfg

        self.stages = ModuleList(
            [
                RepVGGBlock(
                    self.in_channels,
                    self.channels[0],
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
                RepVGGBlock(
                    self.channels[0],
                    self.channels[1],
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                # ConvModule(
                #     self.in_channels,
                #     self.channels[0],
                #     kernel_size=3,
                #     stride=2,
                #     padding=1,
                #     norm_cfg=norm_cfg,
                #     act_cfg=act_cfg),
                # ConvModule(
                #     self.channels[0],
                #     self.channels[1],
                #     kernel_size=3,
                #     stride=2,
                #     padding=1,
                #     norm_cfg=norm_cfg,
                #     act_cfg=act_cfg)
            ]
        )
        # `self.num_shallow_features` is the number of shallow modules in
        # `STDCNet`, which is noted as `Stage1` and `Stage2` in original paper.
        # They are both not used for following modules like Attention
        # Refinement Module and Feature Fusion Module.
        # Thus they would be cut from `outs`. Please refer to Figure 4
        # of original paper for more details.
        self.num_shallow_features = len(self.stages)

        for strides in self.stage_strides:
            idx = len(self.stages) - 1
            self.stages.append(
                self._make_stage(
                    self.channels[idx],
                    self.channels[idx + 1],
                    strides,
                    norm_cfg,
                    act_cfg,
                    bottleneck_type,
                )
            )
        # After appending, `self.stages` is a ModuleList including several
        # shallow modules and STDCModules.
        # (len(self.stages) ==
        # self.num_shallow_features + len(self.stage_strides))
        if self.with_final_conv:
            self.final_conv = ConvModule(
                self.channels[-1],
                max(1024, self.channels[-1]),
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

    def _make_stage(
        self, in_channels, out_channels, strides, norm_cfg, act_cfg, bottleneck_type
    ):
        layers = []
        for i, stride in enumerate(strides):
            layers.append(
                RepSTDCModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride,
                    norm_cfg,
                    act_cfg,
                    num_convs=self.num_convs,
                    fusion_type=bottleneck_type,
                )
            )
        return Sequential(*layers)

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        if self.with_final_conv:
            outs[-1] = self.final_conv(outs[-1])
        outs = outs[self.num_shallow_features :]
        return tuple(outs)

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        self.deploy = True


@MODELS.register_module()
class RepSTDCContextPathNet(BaseModule):
    """STDCNet with Context Path. The `outs` below is a list of three feature
    maps from deep to shallow, whose height and width is from small to big,
    respectively. The biggest feature map of `outs` is outputted for
    `STDCHead`, where Detail Loss would be calculated by Detail Ground-truth.
    The other two feature maps are used for Attention Refinement Module,
    respectively. Besides, the biggest feature map of `outs` and the last
    output of Attention Refinement Module are concatenated for Feature Fusion
    Module. Then, this fusion feature map `feat_fuse` would be outputted for
    `decode_head`. More details please refer to Figure 4 of original paper.

    Args:
        backbone_cfg (dict): Config dict for stdc backbone.
        last_in_channels (tuple(int)), The number of channels of last
            two feature maps from stdc backbone. Default: (1024, 512).
        out_channels (int): The channels of output feature maps.
            Default: 128.
        ffm_cfg (dict): Config dict for Feature Fusion Module. Default:
            `dict(in_channels=512, out_channels=256, scale_factor=4)`.
        upsample_mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``.
        align_corners (str): align_corners argument of F.interpolate. It
            must be `None` if upsample_mode is ``'nearest'``. Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Return:
        outputs (tuple): The tuple of list of output feature map for
            auxiliary heads and decoder head.
    """

    def __init__(
        self,
        backbone_cfg,
        last_in_channels=(1024, 512),
        out_channels=128,
        refine_type=None,
        fusion_type=None,
        ffm_cfg=dict(in_channels=512, out_channels=256, scale_factor=4),
        upsample_mode="nearest",
        align_corners=None,
        norm_cfg=dict(type="BN"),
        init_cfg=None,
        deploy=False,
    ):
        super(RepSTDCContextPathNet, self).__init__(init_cfg=init_cfg)
        self.deploy = deploy
        self.backbone = MODELS.build(backbone_cfg)
        self.arms = ModuleList()
        self.convs = ModuleList()

        arm_module=None
        if refine_type=="CA":
            arm_module=CoordinateAttentionRefinementModule
        elif refine_type=="ECA":
            arm_module=ECAAttentionRefinementModule
        else:
            arm_module=AttentionRefinementModule
            
        for channels in last_in_channels:
            self.arms.append(arm_module(channels, out_channels))
            self.convs.append(
                RepVGGBlock(out_channels, out_channels, norm_cfg=norm_cfg),
                # ConvModule(
                #     out_channels,
                #     out_channels,
                #     3,
                #     padding=1,
                #     norm_cfg=norm_cfg)
            )
        self.conv_avg = ConvModule(
            last_in_channels[0], out_channels, 1, norm_cfg=norm_cfg
        )

        if fusion_type=="CA":
            self.ffm = CoordinateFeatureFusionModule(**ffm_cfg)
        elif fusion_type=="CBAM":
            self.ffm = CBAMFeatureFusionModule(**ffm_cfg)
        else:
            self.ffm = FeatureFusionModule(**ffm_cfg)

        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        if self.deploy:
            self.switch_to_deploy()

    def forward(self, x):
        outs = list(self.backbone(x))
        avg = F.adaptive_avg_pool2d(outs[-1], 1)
        avg_feat = self.conv_avg(avg)

        feature_up = resize(
            avg_feat,
            size=outs[-1].shape[2:],
            mode=self.upsample_mode,
            align_corners=self.align_corners,
        )
        arms_out = []
        for i in range(len(self.arms)):
            x_arm = self.arms[i](outs[len(outs) - 1 - i]) + feature_up
            feature_up = resize(
                x_arm,
                size=outs[len(outs) - 1 - i - 1].shape[2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners,
            )
            feature_up = self.convs[i](feature_up)
            arms_out.append(feature_up)

        feat_fuse = self.ffm(outs[0], arms_out[1])

        # The `outputs` has four feature maps.
        # `outs[0]` is outputted for `STDCHead` auxiliary head.
        # Two feature maps of `arms_out` are outputted for auxiliary head.
        # `feat_fuse` is outputted for decoder head.
        outputs = [outs[0]] + list(arms_out) + [feat_fuse]
        return tuple(outputs)

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
