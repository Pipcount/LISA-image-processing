import matplotlib

matplotlib.use("TkAgg")
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, io, measure, morphology
from skimage.filters import rank
import scipy.ndimage as ndimage


# Load the image
def load_image(index=0):
    dir_path = os.path.join(os.getcwd(), "Images")
    try:
        image_files = [f for f in os.listdir(dir_path) if f.endswith(".tif")]
        if not image_files:
            print("No .tif files found in the Images directory.")
            exit()
        image_path = os.path.join(dir_path, image_files[index])
        image = io.imread(image_path)
    except (FileNotFoundError, IndexError):
        print(f"Error loading image from {dir_path}")
        exit()
    return image[:4096, :4096, 0]


def get_histogram(image):
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.title("Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


def get_balls(image, threshold_value=1):
    binary_image = image < threshold_value
    selem = morphology.disk(3)
    opened_image = morphology.opening(binary_image, selem)

    labeled_image = measure.label(opened_image)

    regions = measure.regionprops(labeled_image)
    centers = [prop.centroid for prop in regions]

    # Create a figure to display the results
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap="gray")

    for i, region in enumerate(regions):
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)

        # Add label text
        center_y, center_x = centers[i]
        ax.text(
            center_x,
            center_y,
            str(i + 1),
            color="yellow",
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_title("Image with Detected and Labeled Balls")
    ax.axis("off")

    plt.show()

    return centers


def get_fibers(image):
    # Store intermediate steps for plotting
    steps = {"Original Image": image}

    # Apply Otsu's threshold to get a binary image
    thresh = filters.threshold_otsu(image)
    binary_fibers = image < thresh  # Fibers are darker
    balls = image < 1
    balls_2 = image > 254
    binary_fibers ^= balls
    binary_fibers ^= balls_2
    steps["1. Otsu Threshold"] = binary_fibers.copy()

    # Remove noise from the image
    binary_fibers = morphology.remove_small_objects(
        binary_fibers, min_size=1500, connectivity=3
    )
    steps["2. Remove Small Objects"] = binary_fibers.copy()

    # 1. Median filter to reduce noise
    denoised = filters.median(binary_fibers, morphology.disk(3))
    steps["3. Median Filter"] = denoised.copy()

    # 2. Morphological opening to clean up
    kernel = morphology.disk(5)
    cleaned = morphology.opening(denoised, kernel)
    steps["4. Morphological Opening"] = cleaned.copy()

    # 3. Remove small components
    labeled = measure.label(cleaned)
    regions = measure.regionprops(labeled)
    final = np.zeros_like(cleaned)
    min_area = 2000

    for region in regions:
        if region.area >= min_area:
            final[labeled == region.label] = 255

    final = morphology.remove_small_objects(final, min_size=min_area)
    steps["5. Final Result"] = final.copy()

    # Plot all steps
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for i, (title, img) in enumerate(steps.items()):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    for i in range(len(steps), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    return binary_fibers


def simple_fiber_detection(image):
    # Store intermediate steps for plotting
    steps = {"Original Image": image.copy()}

    # Gaussian filter to remove salt-and-pepper noise
    filtered = filters.gaussian(image, sigma=1)
    steps["1. Gaussian Filter"] = filtered

    thresh = filters.threshold_otsu(filtered)
    binary_fibers = filtered < thresh  # Fibers are darker
    balls = image < 1
    balls_2 = image > 254
    binary_fibers ^= balls
    binary_fibers ^= balls_2
    binary_fibers = ~binary_fibers
    steps["2. Otsu Threshold"] = binary_fibers.copy()

    binary_fibers = morphology.remove_small_objects(
        binary_fibers, min_size=10000, connectivity=3
    )
    steps["3. Remove Small Objects"] = binary_fibers.copy()

    binary_fibers = morphology.opening(binary_fibers, morphology.disk(3))
    steps["4. Morphological Opening"] = binary_fibers.copy()

    # Use area and aspect ratio to distinguish fibers from noise
    labeled = measure.label(binary_fibers)
    steps["5. Labeled Image"] = labeled

    regions = measure.regionprops(labeled)

    cleaned = np.zeros_like(binary_fibers)

    for region in regions:
        # Fiber criteria: minimum area and elongated shape
        area = region.area
        bbox = region.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        aspect_ratio = max(height, width) / max(min(height, width), 1)

        # Keep regions that are either:
        # - Large enough (likely fiber bundles)
        # - Elongated (likely individual fibers)
        if area > 1000 or (area > 200 and aspect_ratio > 2):
            cleaned[labeled == region.label] = True

    steps["6. Cleaned Fibers"] = cleaned

    # Plot all steps
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for i, (title, img) in enumerate(steps.items()):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    for i in range(len(steps), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    return cleaned


if __name__ == "__main__":
    for i in range(10):
        image = load_image(i)
        # plt.imshow(image, cmap="gray")
        # plt.show()
        # get_histogram(image)
        # centers = get_balls(image.copy())
        fibers = get_fibers(image.copy())
        fibers = simple_fiber_detection(image.copy())
