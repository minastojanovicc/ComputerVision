import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_image(img, color_map=None):
    plt.imshow(img, cmap=color_map)
    plt.axis("off")
    plt.show()


def save_image(img, img_path):
    cv2.imwrite(img_path, img)


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


if __name__ == "__main__":
    # Ucitavanje slike
    img = cv2.imread("./coins/coins.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plot_image(img)

    # Gaussian blur – smanjenje suma
    img = cv2.GaussianBlur(img, (5, 5), 10)

    # Grayscale konverzija
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Segmentacija svih novcica – rucni prag
    _, coins_mask = cv2.threshold(grayscale_img, 180, 255, cv2.THRESH_BINARY_INV)
    plot_image(coins_mask, "gray")

    # Morfologija – popunjavanje rupa
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    coins_mask = cv2.morphologyEx(coins_mask, cv2.MORPH_CLOSE, kernel)
    plot_image(coins_mask, "gray")

    # HSV konverzija
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Segmentacija markera (S kanal)
    _, coin_marker = cv2.threshold(hsv_img[:, :, 1], 75, 255, cv2.THRESH_BINARY)
    plot_image(coin_marker, "gray")

    # Morfologija markera
    coin_marker = cv2.morphologyEx(coin_marker, cv2.MORPH_CLOSE, kernel)
    coin_marker = cv2.morphologyEx(coin_marker, cv2.MORPH_OPEN, kernel)

    # Morfoloska rekonstrukcija
    reconstructed_img = morphological_reconstruction(coin_marker, coins_mask)
    plot_image(reconstructed_img, "gray")

    # Cuvanje rezultata
    save_image(reconstructed_img, "final_mask.png")
