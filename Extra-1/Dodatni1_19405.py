# Mina Stojanovic  19405
import cv2
import numpy as np


def remove_border_objects(mask):

    marker = np.zeros_like(mask)
    marker[0, :] = mask[0, :]
    marker[-1, :] = mask[-1, :]
    marker[:, 0] = mask[:, 0]
    marker[:, -1] = mask[:, -1]

    kernel = np.ones((3, 3), np.uint8)
    border_objects = marker.copy()

    while True:
        dilated = cv2.dilate(border_objects, kernel)
        temp = cv2.bitwise_and(dilated, mask)

        if np.array_equal(temp, border_objects):
            break
        border_objects = temp

    result = cv2.bitwise_xor(mask, border_objects)

    return result


def main():
    input_path = 'slika1.jpg'  # slika2.jpg
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Greška: Nije moguće učitati sliku na putanji '{input_path}'")
        return

    _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    filtered_mask = remove_border_objects(binary_mask)

    output_path = 'result1.jpg' # result2.jpg
    success = cv2.imwrite(output_path, filtered_mask)

    if success:
        print(f"Uspešno sačuvano: {output_path}")
    else:
        print("Greška pri čuvanju slike.")


if __name__ == "__main__":
    main()