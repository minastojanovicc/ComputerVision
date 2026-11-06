import cv2
import numpy as np


def remove_border_objects(mask):
    """
    Uklanja objekte koji dodiruju ivicu slike koristeći
    morfološku rekonstrukciju i logičke operacije.
    """
    # 1. Kreiranje markera (izdvajanje samo piksela na ivicama)
    marker = np.zeros_like(mask)
    marker[0, :] = mask[0, :]  # Gornja ivica
    marker[-1, :] = mask[-1, :]  # Donja ivica
    marker[:, 0] = mask[:, 0]  # Leva ivica
    marker[:, -1] = mask[:, -1]  # Desna ivica

    # 2. Morfološka rekonstrukcija (Uslovna dilatacija)
    # Koristimo 3x3 strukturni element (8-povezanost)
    kernel = np.ones((3, 3), np.uint8)
    border_objects = marker.copy()

    while True:
        # Širimo marker
        dilated = cv2.dilate(border_objects, kernel)
        # Ograničavamo širenje samo na oblasti gde postoji originalna maska
        temp = cv2.bitwise_and(dilated, mask)

        # Provera da li je došlo do promene u ovoj iteraciji
        if np.array_equal(temp, border_objects):
            break
        border_objects = temp

    # 3. Finalno filtriranje: Originalna maska MINUS objekti na ivici
    # XOR operacija efikasno uklanja rekonstruisane objekte
    result = cv2.bitwise_xor(mask, border_objects)

    return result


def main():
    # Učitavanje slike u grayscale modu
    input_path = 'slika2.jpg'
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Greška: Nije moguće učitati sliku na putanji '{input_path}'")
        return

    # Osiguravamo da je slika binarna (Thresholding)
    # Sve što je svetlije od 127 postaje 255 (bela), ostalo 0 (crna)
    _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Poziv funkcije za filtriranje
    filtered_mask = remove_border_objects(binary_mask)

    # Čuvanje rezultata
    output_path = 'result22.jpg'
    success = cv2.imwrite(output_path, filtered_mask)

    if success:
        print(f"Uspešno sačuvano: {output_path}")
    else:
        print("Greška pri čuvanju slike.")


if __name__ == "__main__":
    main()