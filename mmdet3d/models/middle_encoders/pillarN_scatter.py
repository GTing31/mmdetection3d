# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import Tensor, nn

from mmdet3d.registry import MODELS

from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseModule, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseModule, SparseSequential

@MODELS.register_module()
class To_sptensor(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels: int, output_shape: List[int]):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels

    def forward(self,
                voxel_features: Tensor,
                coors: Tensor,
                batch_size: int = None) -> SparseConvTensor:
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases

        coors2d = coors[:, [0, 2, 3]]

        sparse_tensor = SparseConvTensor(
            features=voxel_features,
            indices=coors2d.int(),
            spatial_shape=[self.ny, self.nx],
            batch_size=batch_size)

        # print(f"PointPillarsScatterFeature type: {voxel_features.dtype}, Shape: {voxel_features.shape}")
        # print(f"PointPillarsScatterCoors type: {coors.dtype}, Shape: {coors.shape}")
        # print(f"Middle encoder output type: {type(sparse_tensor)}")
        return sparse_tensor

    # def forward_single(self, voxel_features: Tensor, coors: Tensor,
    #                    batch_size: Tensor) -> SparseConvTensor:
    #     """Scatter features of single sample.
    #
    #     Args:
    #         voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
    #         coors (torch.Tensor): Coordinates of each voxel.
    #             The first column indicates the sample ID.
    #     """
    #     # Create the canvas for this sample
    #     # #生成一个全0的Tensor(畫布)
    #     # canvas = torch.zeros(
    #     #     self.in_channels,
    #     #     self.nx * self.ny,
    #     #     dtype=voxel_features.dtype,
    #     #     device=voxel_features.device)
    #
    #
    #
    #     # indices = coors[:, 2] * self.nx + coors[:, 3]
    #     # indices = indices.long()
    #     # voxels = voxel_features.t()
    #     # # Now scatter the blob back to the canvas.
    #     # canvas[:, indices] = voxels
    #     # Undo the column stacking to final 4-dim tensor
    #
    #
    #     coors2d = coors[:, [0, 2, 3]]
    #
    #     sparse_tensor = SparseConvTensor(
    #         features=voxel_features,
    #         indices=coors2d.int(),
    #         spatial_shape=[self.ny,self.nx],
    #         batch_size=batch_size)
    #
    #     print(f"Middle encoder output type: {type(sparse_tensor)}")
    #     return sparse_tensor
    #
    # def forward_batch(self, voxel_features: Tensor, coors: Tensor,
    #                   batch_size: int) -> SparseConvTensor:
    #     """Scatter features of single sample.
    #
    #     Args:
    #         voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
    #         coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
    #             The first column indicates the sample ID.
    #         batch_size (int): Number of samples in the current batch.
    #     """
    #     # batch_canvas will be the final output.
    #     batch_canvas = []
    #     for batch_itt in range(batch_size):
    #         # Create the canvas for this sample
    #         canvas = torch.zeros(
    #             self.in_channels,
    #             self.nx * self.ny,
    #             dtype=voxel_features.dtype,
    #             device=voxel_features.device)
    #
    #         # Only include non-empty pillars
    #         batch_mask = coors[:, 0] == batch_itt
    #         this_coors = coors[batch_mask, :]
    #         indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
    #         indices = indices.type(torch.long)
    #         voxels = voxel_features[batch_mask, :]
    #         voxels = voxels.t()
    #
    #         # Now scatter the blob back to the canvas.
    #         canvas[:, indices] = voxels
    #
    #         # Append to a list for later stacking.
    #         batch_canvas.append(canvas)
    #
    #     # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
    #     batch_canvas = torch.stack(batch_canvas, 0)
    #
    #     # Undo the column stacking to final 4-dim tensor
    #     batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
    #                                      self.nx)
    #
    #     return batch_canvas
