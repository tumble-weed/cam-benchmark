import torch
import numpy as np
from typing import List, Callable
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

neighbors_weights = [((1, 1), 1 / 12),
                     ((0, 1), 1 / 6),
                     ((-1, 1), 1 / 12),
                     ((1, -1), 1 / 12),
                     ((0, -1), 1 / 6),
                     ((-1, -1), 1 / 12),
                     ((1, 0), 1 / 6),
                     ((-1, 0), 1 / 6)]
class NoisyLinearImputer:
    def __init__(self,
                 noise: float = 0.01,
                 weighting: List[float] = neighbors_weights):
        """
                Noisy linear imputation.
                noise: magnitude of noise to add (absolute, set to 0 for no noise)
                weighting: Weights of the neighboring pixels in the computation.
                List of tuples of (offset, weight)
        """
        self.noise = noise
        self.weighting = neighbors_weights
    perturbation = dutils.TODO

    @staticmethod
    def add_offset_to_indices(indices, offset, mask_shape):
        """ Add the corresponding offset to the indices.
    Return new indices plus a valid bit-vector. """
        cord1 = indices % mask_shape[1]
        cord0 = indices // mask_shape[1]
        cord0 += offset[0]
        cord1 += offset[1]
        valid = ((cord0 < 0) | (cord1 < 0) |
                 (cord0 >= mask_shape[0]) |
                 (cord1 >= mask_shape[1]))
        return ~valid, indices + offset[0] * mask_shape[1] + offset[1]

    @staticmethod
    def setup_sparse_system(mask, img, neighbors_weights):
        """ Vectorized version to set up the equation system.
                mask: (H, W)-tensor of missing pixels.
                Image: (H, W, C)-tensor of all values.
                Return (N,N)-System matrix, (N,C)-Right hand side for each of the C channels.
        """
        maskflt = mask.flatten()
        imgflat = img.reshape((img.shape[0], -1))
    # Indices that are imputed in the flattened mask:
        indices = np.argwhere(maskflt == 0).flatten()
        coords_to_vidx = np.zeros(len(maskflt), dtype=int)
        coords_to_vidx[indices] = np.arange(len(indices))
        numEquations = len(indices)
    # System matrix:
        A = lil_matrix((numEquations, numEquations))
        b = np.zeros((numEquations, img.shape[0]))
    # Sum of weights assigned:
        sum_neighbors = np.ones(numEquations)
        for n in neighbors_weights:
            offset, weight = n[0], n[1]
            # Take out outliers
            valid, new_coords = NoisyLinearImputer.add_offset_to_indices(
                indices, offset, mask.shape)
            valid_coords = new_coords[valid]
            valid_ids = np.argwhere(valid == 1).flatten()
            # Add values to the right hand-side
            has_values_coords = valid_coords[maskflt[valid_coords] > 0.5]
            has_values_ids = valid_ids[maskflt[valid_coords] > 0.5]
            b[has_values_ids, :] -= weight * imgflat[:, has_values_coords].T
            # Add weights to the system (left hand side)
# Find coordinates in the system.
            has_no_values = valid_coords[maskflt[valid_coords] < 0.5]
            variable_ids = coords_to_vidx[has_no_values]
            has_no_values_ids = valid_ids[maskflt[valid_coords] < 0.5]
            A[has_no_values_ids, variable_ids] = weight
            # Reduce weight for invalid
            sum_neighbors[np.argwhere(valid == 0).flatten()] = \
                sum_neighbors[np.argwhere(valid == 0).flatten()] - weight

        A[np.arange(numEquations), np.arange(numEquations)] = -sum_neighbors
        return A, b

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        """ Our linear inputation scheme. """
        """
		This is the function to do the linear infilling
		img: original image (C,H,W)-tensor;
		mask: mask; (H,W)-tensor

		"""
        imgflt = img.reshape(img.shape[0], -1)
        maskflt = mask.reshape(-1)
        # Indices that need to be imputed.
        indices_linear = np.argwhere(maskflt == 0).flatten()
        # Set up sparse equation system, solve system.
        A, b = NoisyLinearImputer.setup_sparse_system(
            mask.numpy(), img.numpy(), neighbors_weights)
        res = torch.tensor(spsolve(csc_matrix(A), b), dtype=torch.float)

        # Fill the values with the solution of the system.
        img_infill = imgflt.clone()
        img_infill[:, indices_linear] = res.t() + self.noise * \
            torch.randn_like(res.t())
        #p46()
        if np.isnan(img_infill).any() or np.isinf(img_infill).any():
            dutils.pause()
        return img_infill.reshape_as(img)
