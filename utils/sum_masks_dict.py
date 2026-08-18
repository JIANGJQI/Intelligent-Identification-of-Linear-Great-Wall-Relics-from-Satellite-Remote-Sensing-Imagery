import numpy as np
from BiSAM_CD import compute_mask_iou_batch


import numpy as np
from BiSAM_CD import compute_mask_iou_batch


def sum_masks_dict(masks_A, masks_B=None, iou_threshold=0.5):
    """
    Merge masks from two dictionaries, removing highly overlapping masks and performing logical OR operation

    This function processes two mask dictionaries, computes their IoU, removes highly overlapping masks,
    and returns a merged mask. When two mask dictionaries are provided, it compares their similarity,
    removes duplicate masks with IoU above the threshold, and merges the remaining masks.

    Args:
        masks_A (dict): First mask dictionary with object IDs as keys and corresponding mask arrays as values
        masks_B (dict, optional): Second mask dictionary with object IDs as keys and corresponding mask arrays as values, defaults to None
        iou_threshold (float): IoU threshold for determining mask duplicates, defaults to 0.5

    Returns:
        numpy.ndarray: Merged mask array with uint8 data type, same shape as input masks
    """
    # Handle empty inputs
    if not masks_A and (masks_B is None or not masks_B):
        # Get reference shape (if unable to get, raise exception or specify default shape)
        try:
            ref_shape = next(iter(masks_A.values())).shape
        except StopIteration:
            ref_shape = (1, 1024, 1024)  # Default shape
        return np.zeros(ref_shape, dtype=np.uint8)

    try:
        merged_mask = np.zeros_like(next(iter(masks_A.values())), dtype=np.uint8)
    except StopIteration:
        ref_shape = (1, 1024, 1024)  # Default shape
        merged_mask = np.zeros(ref_shape, dtype=np.uint8)

    # No masks to compare, return merged mask directly
    if masks_B is None:
        for mask in masks_A.values():
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)
        return merged_mask

    # Convert masks_A and masks_B to NumPy arrays
    mask_array_A = np.array([m > 0 for m in masks_A.values()])
    mask_array_B = np.array([m > 0 for m in masks_B.values()])

    # Compute IoU for all mask pairs
    iou_matrix = compute_mask_iou_batch(mask_array_A, mask_array_B)

    # Find keys that need to be removed
    keys_to_remove = {"A": [], "B": []}
    for idx_A, obj_id_A in enumerate(masks_A.keys()):
        for idx_B, obj_id_B in enumerate(masks_B.keys()):
            if iou_matrix[idx_A, idx_B] > iou_threshold:
                if obj_id_A not in keys_to_remove["A"]:
                    keys_to_remove["A"].append(obj_id_A)
                if obj_id_B not in keys_to_remove["B"]:
                    keys_to_remove["B"].append(obj_id_B)

    # Merge masks from the first dictionary that are not marked as duplicates
    for obj_id, mask in masks_A.items():
        if obj_id not in keys_to_remove["A"]:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)

    # Merge masks from the second dictionary that are not marked as duplicates
    for obj_id, mask in masks_B.items():
        if obj_id not in keys_to_remove["B"]:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)

    return merged_mask
