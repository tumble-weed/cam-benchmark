# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


from __future__ import division
from __future__ import print_function


# __all__ = [
#     "get_masked_input",
#     "Perturbation",
#     "BLUR_PERTURBATION",
#     "FADE_PERTURBATION",
#     "PRESERVE_VARIANT",
#     "DELETE_VARIANT",
#     "DUAL_VARIANT",
# ]

import math
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn.functional as F
from torchray.utils import imsmooth, imsc
import colorful
# from .common import resize_saliency

BLUR_PERTURBATION = "blur"
"""Blur-type perturbation for :class:`Perturbation`."""

FADE_PERTURBATION = "fade"
"""Fade-type perturbation for :class:`Perturbation`."""

PRESERVE_VARIANT = "preserve"
"""Preservation game for :func:`extremal_perturbation`."""

DELETE_VARIANT = "delete"
"""Deletion game for :func:`extremal_perturbation`."""

DUAL_VARIANT = "dual"
"""Combined game for :func:`extremal_perturbation`."""


# class Perturbation:
#     r"""Perturbation pyramid.

#     The class takes as input a tensor :attr:`input` and applies to it
#     perturbation of increasing strenght, storing the resulting pyramid as
#     the class state. The method :func:`apply` can then be used to generate an
#     inhomogeneously perturbed image based on a certain perturbation mask.

#     The pyramid :math:`y` is the :math:`L\times C\times H\times W` tensor

#     .. math::
#         y_{lcvu} = [\operatorname{perturb}(x, \sigma_l)]_{cvu}

#     where :math:`x` is the input tensor, :math:`c` a channel, :math:`vu`,
#     the spatial location, :math:`l` a perturbation level,  and
#     :math:`\operatorname{perturb}` is a perturbation operator.

#     For the *blur perturbation* (:attr:`BLUR_PERTURBATION`), the perturbation
#     operator amounts to convolution with a Gaussian whose kernel has
#     standard deviation :math:`\sigma_l = \sigma_{\mathrm{max}} (1 -  l/ (L-1))`:

#     .. math::
#         \operatorname{perturb}(x, \sigma_l) = g_{\sigma_l} \ast x

#     For the *fade perturbation* (:attr:`FADE_PERTURBATION`),

#     .. math::
#         \operatorname{perturb}(x, \sigma_l) = \sigma_l \cdot x

#     where  :math:`\sigma_l =  l / (L-1)`.

#     Note that in all cases the last pyramid level :math:`l=L-1` corresponds
#     to the unperturbed input and the first :math:`l=0` to the maximally
#     perturbed input.

#     Args:
#         input (:class:`torch.Tensor`): A :math:`1\times C\times H\times W`
#             input tensor (usually an image).
#         num_levels (int, optional): Number of pyramid leves. Defaults to 8.
#         type (str, optional): Perturbation type (:ref:`ep_perturbations`).
#         max_blur (float, optional): :math:`\sigma_{\mathrm{max}}` for the
#             Gaussian blur perturbation. Defaults to 20.

#     Attributes:
#         pyramid (:class:`torch.Tensor`): A :math:`L\times C\times H\times W`
#             tensor with :math:`L` ():attr:`num_levels`) increasingly
#             perturbed versions of the input tensor.
#     """

#     def __init__(self, input, num_levels=8, max_blur=20, type=BLUR_PERTURBATION,input_b=None):
#         self.type = type
#         self.num_levels = num_levels
#         self.pyramid = []
#         self.pyramid_b = []
#         assert num_levels >= 2
#         assert max_blur > 0
#         with torch.no_grad():
#             for sigma in torch.linspace(0, 1, self.num_levels):
#                 if type == BLUR_PERTURBATION:
#                     y = imsmooth(input, sigma=(1 - sigma) * max_blur)
#                     if input_b is not None:
#                         y_b = imsmooth(input_b, sigma=(1 - sigma) * max_blur)
#                 elif type == FADE_PERTURBATION:
#                     print(colorful.red("not implemented for input_b"))
#                     y = input * sigma
#                 else:
#                     assert False
#                 self.pyramid.append(y)
#                 if input_b is not None:
#                     self.pyramid_b.append(y_b)
#             self.pyramid = torch.cat(self.pyramid, dim=0)
#             if input_b is not None:
#                 self.pyramid_b = torch.cat(self.pyramid_b, dim=0)
#     def apply(self, mask):
#         r"""Generate a perturbetd tensor from a perturbation mask.

#         The :attr:`mask` is a tensor :math:`K\times 1\times H\times W`
#         with spatial dimensions :math:`H\times W` matching the input
#         tensor passed upon instantiation of the class. The output
#         is a :math:`K\times C\times H\times W` with :math:`K` perturbed
#         versions of the input tensor, one for each mask.

#         Masks values are in the range 0 to 1, where 1 means that the input
#         tensor is copied as is, and 0 that it is maximally perturbed.

#         Formally, the output is then given by:

#         .. math::
#             z_{kcvu} = y_{m_{k1cu}, c, v, u}

#         where :math:`k` index the mask, :math:`c` the feature channel,
#         :math:`vu` the spatial location, :math:`y` is the pyramid tensor,
#         and :math:`m` the mask tensor :attr:`mask`.

#         The mask must be in the range :math:`[0, 1]`. Linear interpolation
#         is used to index the perturbation level dimension of :math:`y`.

#         Args:
#             mask (:class:`torch.Tensor`): A :math:`K\times 1\times H\times W`
#                 input tensor representing :math:`K` masks.

#         Returns:
#             :class:`torch.Tensor`: A :math:`K\times C\times H\times W` tensor
#             with :math:`K` perturbed versions of the input tensor.
#         """
#         n = mask.shape[0]
#         w = mask.reshape(n, 1, *mask.shape[1:])
#         w = w * (self.num_levels - 1)
#         k = w.floor()
#         w = w - k
#         k = k.long()

#         y = self.pyramid[None, :]
#         y = y.expand(n, *y.shape[1:])
#         k = k.expand(n, 1, *y.shape[2:])
#         y0 = torch.gather(y, 1, k)
        
#         if len(self.pyramid_b):
#             # import ipdb; ipdb.set_trace()
#             # y_b = self.pyramid_b.flip(0)[None, :]
#             y_b = self.pyramid_b[None, :]
#             y_b = y_b.expand(n, *y_b.shape[1:])            
#             y1 = torch.gather(y_b, 1, torch.clamp( (self.num_levels - 1) - k, max=self.num_levels - 1))
#         else:
#             y1 = torch.gather(y, 1, torch.clamp(k + 1, max=self.num_levels - 1))

#         return ((1 - w) * y0 + w * y1).squeeze(dim=1)

#     def to(self, dev):
#         """Switch to another device.

#         Args:
#             dev: PyTorch device.

#         Returns:
#             Perturbation: self.
#         """
#         self.pyramid.to(dev)
#         return self

#     def __str__(self):
#         return (
#             f"Perturbation:\n"
#             f"- type: {self.type}\n"
#             f"- num_levels: {self.num_levels}\n"
#             f"- pyramid shape: {list(self.pyramid.shape)}"
#         )

class Perturbation:
    r"""Perturbation pyramid.

    The class takes as input a tensor :attr:`input` and applies to it
    perturbation of increasing strenght, storing the resulting pyramid as
    the class state. The method :func:`apply` can then be used to generate an
    inhomogeneously perturbed image based on a certain perturbation mask.

    The pyramid :math:`y` is the :math:`L\times C\times H\times W` tensor

    .. math::
        y_{lcvu} = [\operatorname{perturb}(x, \sigma_l)]_{cvu}

    where :math:`x` is the input tensor, :math:`c` a channel, :math:`vu`,
    the spatial location, :math:`l` a perturbation level,  and
    :math:`\operatorname{perturb}` is a perturbation operator.

    For the *blur perturbation* (:attr:`BLUR_PERTURBATION`), the perturbation
    operator amounts to convolution with a Gaussian whose kernel has
    standard deviation :math:`\sigma_l = \sigma_{\mathrm{max}} (1 -  l/ (L-1))`:

    .. math::
        \operatorname{perturb}(x, \sigma_l) = g_{\sigma_l} \ast x

    For the *fade perturbation* (:attr:`FADE_PERTURBATION`),

    .. math::
        \operatorname{perturb}(x, \sigma_l) = \sigma_l \cdot x

    where  :math:`\sigma_l =  l / (L-1)`.

    Note that in all cases the last pyramid level :math:`l=L-1` corresponds
    to the unperturbed input and the first :math:`l=0` to the maximally
    perturbed input.

    Args:
        input (:class:`torch.Tensor`): A :math:`1\times C\times H\times W`
            input tensor (usually an image).
        num_levels (int, optional): Number of pyramid leves. Defaults to 8.
        type (str, optional): Perturbation type (:ref:`ep_perturbations`).
        max_blur (float, optional): :math:`\sigma_{\mathrm{max}}` for the
            Gaussian blur perturbation. Defaults to 20.

    Attributes:
        pyramid (:class:`torch.Tensor`): A :math:`L\times C\times H\times W`
            tensor with :math:`L` ():attr:`num_levels`) increasingly
            perturbed versions of the input tensor.
    """

    def __init__(self, input, num_levels=8, max_blur=20, type=BLUR_PERTURBATION,input_b=None):
        self.type = type
        self.num_levels = num_levels
        self.pyramid = []
        assert num_levels >= 2
        assert max_blur > 0
        with torch.no_grad():
            for sigma in torch.linspace(0, 1, self.num_levels):

                if type == BLUR_PERTURBATION:
                    # if dutils.hack('FOR_COMPILE',default=False,env='FOR_COMPILE'):
                    #     dutils.pause()
                    # y0 is the most blurred version
                    # y = imsmooth(input, sigma=(1 - 0.1429) * max_blur)
                    y = imsmooth(input, sigma=(1 - sigma) * max_blur)
                    
                    # import ipdb; ipdb.set_trace()
                elif type == FADE_PERTURBATION:
                    y = input * sigma
                else:
                    assert False
                self.pyramid.append(y)

            
            self.pyramid = torch.cat(self.pyramid, dim=0)
    @torch.compile
    def apply(self, mask):
        r"""Generate a perturbetd tensor from a perturbation mask.

        The :attr:`mask` is a tensor :math:`K\times 1\times H\times W`
        with spatial dimensions :math:`H\times W` matching the input
        tensor passed upon instantiation of the class. The output
        is a :math:`K\times C\times H\times W` with :math:`K` perturbed
        versions of the input tensor, one for each mask.

        Masks values are in the range 0 to 1, where 1 means that the input
        tensor is copied as is, and 0 that it is maximally perturbed.

        Formally, the output is then given by:

        .. math::
            z_{kcvu} = y_{m_{k1cu}, c, v, u}

        where :math:`k` index the mask, :math:`c` the feature channel,
        :math:`vu` the spatial location, :math:`y` is the pyramid tensor,
        and :math:`m` the mask tensor :attr:`mask`.

        The mask must be in the range :math:`[0, 1]`. Linear interpolation
        is used to index the perturbation level dimension of :math:`y`.

        Args:
            mask (:class:`torch.Tensor`): A :math:`K\times 1\times H\times W`
                input tensor representing :math:`K` masks.

        Returns:
            :class:`torch.Tensor`: A :math:`K\times C\times H\times W` tensor
            with :math:`K` perturbed versions of the input tensor.
        """
        n = mask.shape[0]
        w = mask.reshape(n, 1, *mask.shape[1:])
        # w = mask.view(n, 1, *mask.shape[1:])
        w = w * (self.num_levels - 1)
        k = w.floor()
        w = w - k
        k = k.long()

        y = self.pyramid[None, :]
        y = y.expand(n, *y.shape[1:])
        k = k.expand(n, 1, *y.shape[2:])
        y0 = torch.gather(y, 1, k)
        y1 = torch.gather(y, 1, torch.clamp(k + 1, max=self.num_levels - 1))
        # import ipdb; ipdb.set_trace()
        return ((1 - w) * y0.detach() + w * y1.detach()).squeeze(dim=1)

    def to(self, dev):
        """Switch to another device.

        Args:
            dev: PyTorch device.

        Returns:
            Perturbation: self.
        """
        self.pyramid.to(dev)
        return self

    def __str__(self):
        return (
            f"Perturbation:\n"
            f"- type: {self.type}\n"
            f"- num_levels: {self.num_levels}\n"
            f"- pyramid shape: {list(self.pyramid.shape)}"
        )

def get_masked_input(
                    input,
                    mask_,
                    perturbation=BLUR_PERTURBATION,
                    num_levels=8,

                    variant=PRESERVE_VARIANT,
                    smooth=0,
                    input_b = None,
                    ):
    r"""Compute a set of extremal perturbations.

    The function takes a :attr:`model`, an :attr:`input` tensor :math:`x`
    of size :math:`1\times C\times H\times W`, and a :attr:`target`
    activation channel. It produces as output a
    :math:`K\times C\times H\times W` tensor where :math:`K` is the number
    of specified :attr:`areas`.

    Each mask, which has approximately the specified area, is searched
    in order to maximise the (spatial average of the) activations
    in channel :attr:`target`. Alternative objectives can be specified
    via :attr:`reward_func`.

    Args:
        model (:class:`torch.nn.Module`): model.
        input (:class:`torch.Tensor`): input tensor.
        target (int): target channel.
        areas (float or list of floats, optional): list of target areas for saliency
            masks. Defaults to `[0.1]`.
        perturbation (str, optional): :ref:`ep_perturbations`.
        max_iter (int, optional): number of iterations for optimizing the masks.
        num_levels (int, optional): number of buckets with which to discretize
            and linearly interpolate the perturbation
            (see :class:`Perturbation`). Defaults to 8.
        step (int, optional): mask step (see :class:`MaskGenerator`).
            Defaults to 7.
        sigma (float, optional): mask smoothing (see :class:`MaskGenerator`).
            Defaults to 21.
        jitter (bool, optional): randomly flip the image horizontally at each iteration.
            Defaults to True.
        variant (str, optional): :ref:`ep_variants`. Defaults to
            :attr:`PRESERVE_VARIANT`.
        print_iter (int, optional): frequency with which to print losses.
            Defaults to None.
        debug (bool, optional): If True, generate debug plots.
        reward_func (function, optional): function that generates reward tensor
            to backpropagate.
        resize (bool, optional): If True, upsamples the masks the same size
            as :attr:`input`. It is also possible to specify a pair
            (width, height) for a different size. Defaults to False.
        resize_mode (str, optional): Upsampling method to use. Defaults to
            ``'bilinear'``.
        smooth (float, optional): Apply Gaussian smoothing to the masks after
            computing them. Defaults to 0.

    Returns:
        A tuple containing the masks and the energies.
        The masks are stored as a :class:`torch.Tensor`
        of dimension
    """


    device = input.device

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    if isinstance(perturbation,str):
        perturbation = Perturbation(
            input,
            num_levels=num_levels,
            type=perturbation,
            # input_b= input_b
        ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]


    # Apply the mask.
    if variant == DELETE_VARIANT:
        x = perturbation.apply(1 - mask_)
    elif variant == PRESERVE_VARIANT:
        x = perturbation.apply(mask_)
    elif variant == DUAL_VARIANT:
        x = torch.cat((
            perturbation.apply(mask_),
            perturbation.apply(1 - mask_),
        ), dim=0)
    else:
        assert False
    return x,perturbation


def laplacian_pyramid_blending(img1, img2, mask, num_levels=6):
    # Generate Gaussian pyramid for image 1, image 2 and mask
    G1 = img1.float()
    G2 = img2.float()
    GM = mask.float()

    gp1 = [G1]
    gp2 = [G2]
    gpM = [GM]

    for i in range(num_levels):
        # G1 = F.avg_pool2d(G1, kernel_size=2)
        # G2 = F.avg_pool2d(G2, kernel_size=2)
        # GM = F.avg_pool2d(GM, kernel_size=2)
        G1 = F.interpolate(G1,size=(G1.shape[2]//2,G1.shape[3]//2),antialias=True,mode='bilinear')
        G2 = F.interpolate(G2,size=(G2.shape[2]//2,G2.shape[3]//2),antialias=True,mode='bilinear')
        GM = F.interpolate(GM,size=(GM.shape[2]//2,GM.shape[3]//2),antialias=True,mode='bilinear')

        gp1.append(G1)
        gp2.append(G2)
        gpM.append(GM)

    # Generate Laplacian Pyramid for image 1, image 2 and mask
    lp1 = [gp1[num_levels - 1]]
    lp2 = [gp2[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]

    for i in range(num_levels - 1, 0, -1):
        size = (gp1[i - 1].shape[2], gp1[i - 1].shape[3])

        L1 = gp1[i - 1] - F.interpolate(gp1[i], size=size, mode='bilinear', align_corners=False)
        L2 = gp2[i - 1] - F.interpolate(gp2[i], size=size, mode='bilinear', align_corners=False)
        lp1.append(L1)
        lp2.append(L2)

        # Combine the two Laplacian images using the mask
        GM = F.interpolate(gpM[num_levels - i], size=size, mode='bilinear', align_corners=False)
        LMR = 1 - GM
        lpMr = lp1[num_levels - i] * GM + lp2[num_levels - i] * LMR
        gpMr.append(lpMr)

    # Reconstruct the final image using the Laplacian pyramid
    gpMr = list(reversed(gpMr))
    final_img = gpMr[num_levels - 1]
    # import ipdb; ipdb.set_trace()
    for i in range(num_levels - 1, 0, -1):
        size = (gp1[i - 1].shape[2], gp1[i - 1].shape[3])
        final_img = F.interpolate(final_img, size=size, mode='bilinear', align_corners=False) + gpMr[i - 1]

    return final_img


class MaskGenerator:
    r"""Mask generator.

    The class takes as input the mask parameters and returns
    as output a mask.

    Args:
        shape (tuple of int): output shape.
        step (int): parameterization step in pixels.
        sigma (float): kernel size.
        clamp (bool, optional): whether to clamp the mask to [0,1]. Defaults to True.
        pooling_mehtod (str, optional): `'softmax'` (default),  `'sum'`, '`sigmoid`'.

    Attributes:
        shape (tuple): the same as the specified :attr:`shape` parameter.
        shape_in (tuple): spatial size of the parameter tensor.
        shape_out (tuple): spatial size of the output mask including margin.
    """

    def __init__(self, shape, step, sigma, clamp=True, pooling_method='softmax'):
        self.shape = shape
        self.step = step
        self.sigma = sigma
        self.coldness = 20
        self.clamp = clamp
        self.pooling_method = pooling_method

        assert int(step) == step

        # self.kernel = lambda z: (z < 1).float()
        self.kernel = lambda z: torch.exp(-2 * ((z - .5).clamp(min=0)**2))

        self.margin = self.sigma
        # self.margin = 0
        self.padding = 1 + math.ceil((self.margin + sigma) / step)
        self.radius = 1 + math.ceil(sigma / step)
        self.shape_in = [math.ceil(z / step) for z in self.shape]
        self.shape_mid = [
            z + 2 * self.padding - (2 * self.radius + 1) + 1
            for z in self.shape_in
        ]
        self.shape_up = [self.step * z for z in self.shape_mid]
        self.shape_out = [z - step + 1 for z in self.shape_up]

        self.weight = torch.zeros((
            1,
            (2 * self.radius + 1)**2,
            self.shape_out[0],
            self.shape_out[1]
        ))

        step_inv = [
            torch.tensor(zm, dtype=torch.float32) /
            torch.tensor(zo, dtype=torch.float32)
            for zm, zo in zip(self.shape_mid, self.shape_up)
        ]

        for ky in range(2 * self.radius + 1):
            for kx in range(2 * self.radius + 1):
                uy, ux = torch.meshgrid(
                    torch.arange(self.shape_out[0], dtype=torch.float32),
                    torch.arange(self.shape_out[1], dtype=torch.float32)
                )
                iy = torch.floor(step_inv[0] * uy) + ky - self.padding
                ix = torch.floor(step_inv[1] * ux) + kx - self.padding

                delta = torch.sqrt(
                    (uy - (self.margin + self.step * iy))**2 +
                    (ux - (self.margin + self.step * ix))**2
                )

                k = ky * (2 * self.radius + 1) + kx

                self.weight[0, k] = self.kernel(delta / sigma)

    def generate(self, mask_in):
        r"""Generate a mask.

        The function takes as input a parameter tensor :math:`\bar m` for
        :math:`K` masks, which is a :math:`K\times 1\times H_i\times W_i`
        tensor where `H_i\times W_i` are given by :attr:`shape_in`.

        Args:
            mask_in (:class:`torch.Tensor`): mask parameters.

        Returns:
            tuple: a pair of mask, cropped and full. The cropped mask is a
            :class:`torch.Tensor` with the same spatial shape :attr:`shape`
            as specfied upon creating this object. The second mask is the same,
            but with an additional margin and shape :attr:`shape_out`.
        """

        mask = F.unfold(mask_in,
                        (2 * self.radius + 1,) * 2,
                        padding=(self.padding,) * 2)
        mask = mask.reshape(
            len(mask_in), -1, self.shape_mid[0], self.shape_mid[1])
        # mask = mask.view(
        #     len(mask_in), -1, self.shape_mid[0], self.shape_mid[1])
        mask = F.interpolate(mask, size=self.shape_up, mode='nearest')
        mask = F.pad(mask, (0, -self.step + 1, 0, -self.step + 1))
        mask = self.weight * mask

        if self.pooling_method == 'sigmoid':
            if self.coldness == float('+Inf'):
                mask = (mask.sum(dim=1, keepdim=True) - 5 > 0).float()
            else:
                mask = torch.sigmoid(
                    self.coldness * mask.sum(dim=1, keepdim=True) - 3
                )
        elif self.pooling_method == 'softmax':
            if self.coldness == float('+Inf'):
                mask = mask.max(dim=1, keepdim=True)[0]
            else:
                mask = (
                    mask * F.softmax(self.coldness * mask, dim=1)
                ).sum(dim=1, keepdim=True)

        elif self.pooling_method == 'sum':
            mask = mask.sum(dim=1, keepdim=True)
        else:
            assert False, f"Unknown pooling method {self.pooling_method}"
        m = round(self.margin)
        
        if self.clamp:
            # dutils.pause()
            mask = mask.clamp(min=0, max=1)
        cropped = mask[:, :, m:m + self.shape[0], m:m + self.shape[1]]
        return cropped, mask

    def to(self, dev):
        """Switch to another device.

        Args:
            dev: PyTorch device.

        Returns:
            MaskGenerator: self.
        """
        self.weight = self.weight.to(dev)
        return self


if __name__ == '__main__':
    import torchvision
    input = torch.ones(1,3,224,224)
    mask = torch.ones(1,1,224,224)
    model = torchvision.models.vgg16(pretrained=True)
    masked_input = get_masked_input(model,
                    input,
                    mask,
                    perturbation=BLUR_PERTURBATION,
                    num_levels=8,
                    jitter=True,
                    variant=PRESERVE_VARIANT,
                    smooth=0)