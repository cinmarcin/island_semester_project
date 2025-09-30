from nilearn.glm.second_level import SecondLevelModel
import numpy as np

def check_second_level_mask(second_level_model):
    mask = second_level_model.masker_.mask_img_
    mask_data = mask.get_fdata()
    print(f"Mask shape: {mask.shape}")
    print(f"Mask non-zero voxels: {np.sum(mask_data>0)}")

def check_images(images, name="Image"):
    for i, img in enumerate(images):
        data = img.get_fdata()
        print(f"{name} {i}: shape={data.shape}, min={np.min(data):.3f}, max={np.max(data):.3f}, mean={np.mean(data):.3f}")