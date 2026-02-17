import cv2
import numpy as np
import os


def crop_cells(
    brightfield_img,
    hologram_img,
    patch_size=256,
    min_area=2000,
    max_area=10000,
    min_circularity=0.1,
    filename=None,
    save_dir=None
):
    """
    Segment RBC cells from brightfield image and extract centered patches
    from both brightfield and hologram images.

    Parameters
    ----------
    brightfield_img : np.ndarray
        Input brightfield image (BGR or grayscale).
    hologram_img : np.ndarray
        Corresponding hologram image (aligned with brightfield).
    patch_size : int
        Size of square patch (default 256).
    min_area : int
        Minimum contour area to accept as RBC.
    max_area : int
        Maximum contour area to accept as RBC.
    min_circularity : float
        Minimum circularity threshold.
    filepaths : tuple/list of str or None
        If provided, patches will be saved to these paths.

    Returns
    -------
    holo_patches : list of np.ndarray
    bright_patches : list of np.ndarray
    centers : list of (cx, cy)
    mask : segmentation mask (for debugging)
    """

    # Ensure grayscale
    if len(brightfield_img.shape) == 3:
        gray = cv2.cvtColor(brightfield_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = brightfield_img.copy()

    # 1️⃣ Illumination correction
    background = cv2.GaussianBlur(gray, (101, 101), 0)
    corrected = cv2.subtract(gray, background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    # 2️⃣ Edge detection
    blur = cv2.GaussianBlur(corrected, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    # 3️⃣ Close edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4️⃣ Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holo_patches = []
    bright_patches = []
    centers = []

    mask = np.zeros(gray.shape, dtype=np.uint8)
    half = patch_size // 2

    count = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Reject edge cells
        if (
            cx - half < 0 or
            cy - half < 0 or
            cx + half >= gray.shape[1] or
            cy + half >= gray.shape[0]
        ):
            continue

        # Extract patches
        holo_patch = hologram_img[cy-half:cy+half, cx-half:cx+half]
        bright_patch = brightfield_img[cy-half:cy+half, cx-half:cx+half]

        holo_patches.append(holo_patch)
        bright_patches.append(bright_patch)
        centers.append((cx, cy))

        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

        if filename is not None and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            bf_dir_path = os.path.join(save_dir, "brightfield")
            os.makedirs(bf_dir_path, exist_ok=True)

            holo_dir_path = os.path.join(save_dir, "hologram")
            os.makedirs(holo_dir_path, exist_ok=True)

            cv2.imwrite(os.path.join(bf_dir_path, f"{filename}_{count}.png"), bright_patch)
            cv2.imwrite(os.path.join(holo_dir_path, f"{filename}_{count}.png"), holo_patch)

        count += 1
    if filename is not None and save_dir is not None:
        mask_dir_path = os.path.join(save_dir, "mask")
        os.makedirs(mask_dir_path, exist_ok=True)
        cv2.imwrite(os.path.join(mask_dir_path, f"{filename}_mask.png"), mask)

    return holo_patches, bright_patches, centers, mask
