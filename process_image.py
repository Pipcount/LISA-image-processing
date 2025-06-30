import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage import morphology, measure, io
import os


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


def get_balls(image, threshold_value=1):
    binary_image = image < threshold_value
    selem = morphology.disk(3)
    opened_image = morphology.opening(binary_image, selem)

    labeled_image = measure.label(opened_image)
    plt.imshow(labeled_image, cmap="gray")
    plt.show()

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

    ax.set_title("Image with Detected and Labeled Balls")
    ax.axis("off")

    plt.show()

    return centers


def get_histogram(image):
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.show()


if __name__ == "__main__":
    image = load_image()
    # get_histogram(image)
    centers = get_balls(image.copy())
    print(centers)
