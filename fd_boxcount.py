import numpy as np
import cv2


def fd_std_boxcount(img_bgr, scales=(2, 4, 8, 16, 32, 64)):
    """
    Standard-deviation box-counting (2D intensity field):
    - Convert BGR to grayscale (float32)
    - For each box size h, split image into h x h blocks
    - Compute per-block std; nh = std / h; sum over all blocks -> N(h)
    - Fit log N(h) vs log h; FD = |slope| clipped to [2,3]

    Returns:
        fd (float), valid_scales (np.ndarray), Nh_values (np.ndarray), log_h, log_Nh, coeffs
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    Nh_vals = []
    valid_scales = []
    for h in scales:
        Hc = (H // h) * h
        Wc = (W // h) * h
        if Hc < h or Wc < h:
            continue
        crop = gray[:Hc, :Wc]
        # reshape to (Hc//h, h, Wc//h, h) -> (Hc//h, Wc//h, h, h)
        blocks = crop.reshape(Hc // h, h, Wc // h, h).transpose(0, 2, 1, 3)
        mean_blk = blocks.mean(axis=(2, 3))
        sq_mean = (blocks ** 2).mean(axis=(2, 3))
        std_blk = np.sqrt(np.maximum(0.0, sq_mean - mean_blk ** 2))
        nh = std_blk / float(h)
        Nh_vals.append(float(nh.sum()) + 1e-12)
        valid_scales.append(h)

    if len(valid_scales) < 3:
        return None, np.array(scales), np.array([1] * len(scales)), None, None, None

    log_h = np.log(np.array(valid_scales, dtype=np.float64))
    log_Nh = np.log(np.array(Nh_vals, dtype=np.float64))
    coeffs = np.polyfit(log_h, log_Nh, 1)
    slope = coeffs[0]
    fd = abs(slope)
    fd = float(np.clip(fd, 2.0, 3.0))
    return fd, np.array(valid_scales), np.array(Nh_vals), log_h, log_Nh, coeffs


def fd_3d_dbc(img_bgr, scales=None, max_size=256):
    """
    Differential Box Counting (DBC) on grayscale-as-height (pseudo-3D):
    - Convert BGR to grayscale -> normalize to [0,1] height
    - Quantize height per box of size r using step G = 1/r
    - Count boxes per block: n_r = ceil(max/G) - floor(min/G); sum -> N(r)
    - Fit log N(r) vs log r; FD = 3 - |slope| clipped to [2,3]

    Returns:
        fd3 (float), valid_sizes (np.ndarray), counts (np.ndarray), log_r, log_counts, coeffs
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    # optional downscale for speed
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
        gray = cv2.resize(gray, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        H, W = gray.shape

    if scales is None:
        max_box = max(2, min(H, W) // 4)
        scales = []
        s = 2
        while s <= max_box:
            scales.append(s)
            s *= 2
        if len(scales) < 3:
            scales = [2, 4, 8, 16]

    counts = []
    arr = gray / 255.0
    for r in scales:
        nh = (H // r)
        nw = (W // r)
        if nh < 1 or nw < 1:
            counts.append(0)
            continue
        Hc = nh * r
        Wc = nw * r
        crop = arr[:Hc, :Wc]
        blocks = crop.reshape(nh, r, nw, r).transpose(0, 2, 1, 3)
        bmin = blocks.min(axis=(2, 3))
        bmax = blocks.max(axis=(2, 3))
        G = max(0.001, 1.0 / r)
        l = np.floor(bmin / G)
        k = np.ceil(bmax / G)
        nr = (k - l).astype(np.int32)
        nr = np.maximum(nr, 1)
        counts.append(int(nr.sum()))

    valid_sizes = []
    valid_counts = []
    for s, c in zip(scales, counts):
        if c > 0:
            valid_sizes.append(s)
            valid_counts.append(c)

    if len(valid_counts) < 3:
        return None, np.array(scales), np.array(counts), None, None, None

    log_r = np.log(np.array(valid_sizes, dtype=np.float64))
    log_counts = np.log(np.array(valid_counts, dtype=np.float64))
    coeffs = np.polyfit(log_r, log_counts, 1)
    slope = coeffs[0]
    fd3 = 3.0 - abs(slope)
    fd3 = float(np.clip(fd3, 2.0, 3.0))
    return fd3, np.array(valid_sizes), np.array(valid_counts), log_r, log_counts, coeffs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fd_boxcount.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    # Robust load for non-ASCII paths
    try:
        with open(path, 'rb') as f:
            buf = f.read()
        arr = np.frombuffer(buf, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        img = cv2.imread(path)
    if img is None:
        print("Failed to load image:", path)
        sys.exit(1)
    fd2, s2, n2, _, _, _ = fd_std_boxcount(img)
    fd3, s3, n3, _, _, _ = fd_3d_dbc(img)
    print(f"FD (std-boxcount 2D): {fd2}")
    print(f"FD (DBC pseudo-3D): {fd3}")
