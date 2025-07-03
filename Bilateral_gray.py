import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian(d, sigma):
    return np.exp(-(d ** 2) / (2 * sigma ** 2))

def psnr(f, h):
    mse = np.mean((f.astype(np.float32) - h.astype(np.float32)) ** 2)
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def bilateral_filter_gray(f, window_size, sigma_d, sigma_r):
    half_w = window_size // 2
    X, Y = np.meshgrid(np.arange(-half_w, half_w + 1), np.arange(-half_w, half_w + 1))
    c = gaussian(np.sqrt(Y ** 2 + X ** 2), sigma_d)

    h = np.zeros_like(f, dtype=np.float32)
    f_padded = cv2.copyMakeBorder(f, half_w, half_w, half_w, half_w, cv2.BORDER_REFLECT)

    for i in range(half_w, f_padded.shape[0] - half_w):
        for j in range(half_w, f_padded.shape[1] - half_w):
            f_xi = f_padded[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1].astype(np.float32)
            f_x = f_padded[i, j].astype(np.float32)

            s = gaussian(np.abs(f_xi - f_x), sigma_r)

            w = c * s

            k_x = np.sum(w)

            h_val = np.sum(w * f_xi) / (k_x if k_x != 0 else 1e-5)
            h[i - half_w, j - half_w] = h_val

    return np.clip(h, 0, 255).astype(np.uint8)

# === LOAD GRAYSCALE IMAGE ===
path = "D:/ThucHanh/XuLyAnh/TruocAI/Lena.png"   #Change this path to your grayscale input image
f = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if f is None:
    raise FileNotFoundError(f"Image not found at: {path}")

# === ADD NOISE ===
noise = np.random.normal(0, 25, f.shape).astype(np.float32)
f_noisy = np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# === BILATERAL FILTER ===
sigma_d = 5
sigma_r = 40
window_size = 2 * sigma_d + 1

h = bilateral_filter_gray(f_noisy, window_size, sigma_d, sigma_r)

# === PSNR CALCULATION ===
psnr_noisy = psnr(f, f_noisy)
psnr_filtered = psnr(f, h)

# === DISPLAY RESULTS ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(f, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title(f"Noisy Image\nPSNR: {psnr_noisy:.2f} dB")
plt.imshow(f_noisy, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"After Bilateral Filtering\nPSNR: {psnr_filtered:.2f} dB")
plt.imshow(h, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
