import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian(d, sigma):
    return np.exp(-(d ** 2) / (2 * sigma ** 2))

def psnr(f, h):
    mse = np.mean((f.astype(np.float32) - h.astype(np.float32)) ** 2)
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def bilateral_filter_color(f, window_size, sigma_d, sigma_r):
    half_w = window_size // 2
    X, Y = np.meshgrid(np.arange(-half_w, half_w + 1), np.arange(-half_w, half_w + 1))
    c = gaussian(np.sqrt(Y ** 2 + X ** 2), sigma_d)

    h_img, w_img, _ = f.shape
    h = np.zeros_like(f, dtype=np.float32)
    f_padded = cv2.copyMakeBorder(f, half_w, half_w, half_w, half_w, cv2.BORDER_REFLECT)

    for i in range(half_w, h_img + half_w):
        for j in range(half_w, w_img + half_w):
            f_xi = f_padded[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1, :].astype(np.float32)
            f_x = f_padded[i, j, :].astype(np.float32)

            diff2 = np.sum((f_xi - f_x) ** 2, axis=2)
            s = gaussian(np.sqrt(diff2), sigma_r)

            w = c * s

            k_x = np.sum(w)

            h_pixel = np.sum(f_xi * w[:, :, np.newaxis], axis=(0, 1)) / (k_x if k_x != 0 else 1e-5)
            h[i - half_w, j - half_w, :] = h_pixel

    return np.clip(h, 0, 255).astype(np.uint8)

# === LOAD COLOR IMAGE ===
path = "D:/ThucHanh/XuLyAnh/TruocAI/huou.jpg"   #Change this to the path of your input image file
img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"Image not found at: {path}")

f = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === ADD NOISE ===
noise = np.random.normal(0, 25, f.shape).astype(np.float32)
f_noisy = np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# === BILATERAL FILTER ===
sigma_d = 5
sigma_r = 40
window_size = 2 * sigma_d + 1

h = bilateral_filter_color(f_noisy, window_size, sigma_d, sigma_r)

# === PSNR CALCULATION ===
psnr_noisy = psnr(f, f_noisy)
psnr_filtered = psnr(f, h)

# === DISPLAY RESULTS ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(f)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title(f"Noisy Image\nPSNR: {psnr_noisy:.2f} dB")
plt.imshow(f_noisy)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Image after Bilateral Filter\nPSNR: {psnr_filtered:.2f} dB")
plt.imshow(h)
plt.axis("off")
plt.tight_layout()
plt.show()
