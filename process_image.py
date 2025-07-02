import matplotlib

matplotlib.use("TkAgg")
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, io, measure, morphology
from skimage.filters import frangi, difference_of_gaussians
from skimage.morphology import disk
from skimage.exposure import equalize_adapthist, rescale_intensity


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

    # Big closing to remove noise
    binary_fibers = morphology.closing(binary_fibers, morphology.disk(5))
    steps["4. Morphological Closing"] = binary_fibers.copy()

    binary_fibers = ~binary_fibers

    # Label and remove small components
    labeled = measure.label(binary_fibers)
    regions = measure.regionprops(labeled)
    final = np.zeros_like(binary_fibers)
    min_area = 10000
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


def get_fibers_frangi(image):
    # Store intermediate steps for plotting
    steps = {"Original Image": image.copy()}

    # Frangi filter to detect long fibers
    fibers = frangi(image)
    steps["1. Frangi Filter"] = fibers.copy()

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

    return fibers


def get_fibers_difference_of_gaussians(image):
    # Store intermediate steps for plotting
    steps = {"Original Image": image.copy()}

    # Difference of Gaussians filter to detect long fibers
    fibers = difference_of_gaussians(image, low_sigma=0, high_sigma=100)

    # Normalize the image to the range 0-255 and convert to uint8
    fibers_normalized = (fibers - fibers.min()) / (fibers.max() - fibers.min()) * 255
    fibers = fibers_normalized.astype(np.uint8)
    steps["1. Difference of Gaussians Filter"] = fibers.copy()

    # # fibers - image
    # fibers = fibers - image
    # steps["2. Subtract Original Image"] = fibers.copy()

    # Threshold the image
    thresh = filters.threshold_otsu(fibers)
    binary_fibers = fibers < thresh  # Fibers are darker
    balls = image < 1
    balls_2 = image > 254
    binary_fibers ^= balls
    binary_fibers ^= balls_2
    binary_fibers = ~binary_fibers
    steps["3. Threshold"] = binary_fibers.copy()

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


def preprocess_image(image):
    steps = {"Original Image": image.copy()}

    # reversed = 255 - image
    # steps["1. Reversed Image"] = reversed.copy()

    filtered = filters.gaussian(image, sigma=1)
    steps["1. Gaussian Filter"] = filtered
    # filtered = filters.median(image, morphology.disk(3))
    # steps["2. Median Filter"] = filtered

    # Autolevel
    p2, p98 = np.percentile(filtered, [2, 98])
    autoleveled = rescale_intensity(filtered, in_range=(p2, p98))
    steps["2. Autolevel"] = autoleveled.copy()

    autoleveled = np.uint8(autoleveled * 255)
    equalized = equalize_adapthist(autoleveled, clip_limit=1, kernel_size=400)
    equalized = np.uint8(equalized * 255)
    steps["3. Equalize Adaptive"] = equalized.copy()

    # # Threshold the image
    # thresh = filters.threshold_otsu(autoleveled)
    # binary_fibers = autoleveled < thresh
    # balls = image < 1
    # balls_2 = image > 254
    # binary_fibers ^= balls
    # binary_fibers ^= balls_2
    # steps["5. Threshold"] = binary_fibers.copy()
    #
    # # Close the image
    # kernel = morphology.disk(3)
    # closed = morphology.closing(binary_fibers, kernel)
    # steps["6. Closing"] = closed.copy()
    #
    # # remove small objects
    # closed = morphology.remove_small_objects(closed, min_size=16384)
    # steps["7. Remove Small Objects"] = closed.copy()
    #
    # labeled = measure.label(closed)
    # steps["7. Labeled"] = labeled.copy()

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

    return equalized


def gradient(image):
    # Store intermediate steps for plotting
    steps = {"Original Image": image.copy()}
    # Gradient
    grad = filters.rank.gradient(image, disk(1))
    steps["1. Gradient"] = grad.copy()

    # Threshold the image
    thresh = filters.threshold_otsu(grad)
    binary_fibers = grad < thresh  # Fibers are darker
    balls = image < 5
    balls_2 = image > 254
    binary_fibers ^= balls
    binary_fibers ^= balls_2
    steps["2. Otsu Threshold"] = binary_fibers.copy()

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
    return grad


if __name__ == "__main__":
    for i in range(0, 10):
        image = load_image(i)
        # plt.imshow(image, cmap="gray")
        # plt.show()
        # get_histogram(image)
        # centers = get_balls(image.copy())
        filtered = preprocess_image(image.copy())
        grad = gradient(filtered.copy())
        # fibers = get_fibers(filtered.copy())
        # fibers = simple_fiber_detection(filtered.copy())
        # fibers = get_fibers_frangi(filtered.copy())
        # fibers = get_fibers_difference_of_gaussians(filtered.copy())
