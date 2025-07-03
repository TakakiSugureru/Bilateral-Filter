# Bilateral Filtering (Gray & Color Images)

This project provides a simple implementation of bilateral filtering for grayscale and color images.  
It was rebuilt as a learning exercise based on the paper by **Tomasi and Manduchi (1998)**.

## Original Reference

- Paper: [Bilateral Filtering for Gray and Color Images](https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf)

---
## Files

- `bilateral_gray.py`: Bilateral filter for grayscale images.
- `bilateral_color.py`: Bilateral filter for RGB color images.

## Requirements

- NumPy
- OpenCV (`cv2`)
- Matplotlib

You can install the requirements via pip:

```bash
pip install numpy opencv-python matplotlib
```

## Conclusion

It works well for Gaussian noise, especially on color images, but is less effective for salt-and-pepper noise.

**Advantages:**  
- Keeps edges sharp  
- Easy to implement  

**Disadvantages:**  
- Requires more computation than basic filters
- Not effective for salt-and-pepper noise (use median filter instead)
