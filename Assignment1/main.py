import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft(img):
    img_fft = np.fft.fft2(img)  # pretvaranje slike u frekventni domen (FFT - Fast Fourier Transform), fft2 je jer je u 2 dimenzije
    img_fft = np.fft.fftshift(img_fft)  # pomeranje koordinatnog pocetka u centar slike
    return img_fft

def inverse_fft(magnitude_log, complex_pha):
    img_fft = complex_pha * np.exp(magnitude_log) # vracamo magnitudu iz logaritma i mnozimo sa kompleksnim brojevima na slici
    img_filtered = np.abs(np.fft.ifft2(img_fft)) # funkcija ifft2 vraca sliku iz frekventnog u prostorni domen, nije potrebno raditi ifftshift jer to se implicitno izvrsava

    return img_filtered

def low_pass_filter(img, center, radius):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_pha = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    plt.imshow(img_fft_log)
    plt.show()

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) > radius*radius: # jednacina kruga, zelimo da sve izvan kruga bude nula sto predstavlja idealni low pass filter
                img_fft_log[x,y] = 0

    plt.imshow(img_fft_log)
    plt.show()

    img_filtered = inverse_fft(img_fft_log, img_pha)
    return img_filtered

def standardno(img):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_pha = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)

    plt.imshow(img_fft_log)
    plt.show()

    #276 356
    #236 356
    #276 156
    #236 156

    img_fft_log[356][276] = 0
    img_fft_log[356][236] = 0
    img_fft_log[156][276] = 0
    img_fft_log[156][236] = 0


    plt.imshow(img_fft_log)
    plt.show()

    plt.imshow(inverse_fft(img_fft_log, img_pha), cmap='gray')
    plt.show()

def poboljsano1(img, center, radius):
    img_filtered = low_pass_filter(img,center, radius)
    return img_filtered

def poboljsano2(img, razlika):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_pha = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    plt.imshow(img_fft_log)
    plt.show()

    x = 0
    y = 0
    prosek = (img_fft_log[x][y+1] + img_fft_log[x+1][y+1] + img_fft_log[x+1][y])/3 # (0,0) gornji levi cosak
    if img_fft_log[x][y]-prosek>razlika:
        img_fft_log[x][y] = prosek
        img_fft_log[img.shape[0]-x][img.shape[1]-y] = prosek
    for y in range(1, img.shape[1]-2):
        prosek = (img_fft_log[x][y-1] + img_fft_log[x][y+1] + img_fft_log[x+1][y-1] + img_fft_log[x+1][y] + img_fft_log[x+1][y+1])/5 #gornja ivica
        if img_fft_log[x][y]-prosek>razlika:
            img_fft_log[x][y] = prosek
            img_fft_log[img.shape[0]-x][img.shape[1]-y] = prosek
    y = img.shape[1]-1
    prosek = (img_fft_log[x][y-1] + img_fft_log[x+1][y-1] + img_fft_log[x+1][y])/3 #gornji desni cosak (0,max)
    if img_fft_log[x][y]-prosek>razlika:
        img_fft_log[x][y] = prosek
        img_fft_log[img.shape[0]-x][img.shape[1]-y] = prosek
    for x in range(1, img.shape[0]//2):
        y = 0
        prosek = (img_fft_log[x-1][y] + img_fft_log[x-1][y+1] + img_fft_log[x][y+1] + img_fft_log[x+1][y] + img_fft_log[x+1][y+1])/5 #leva ivica
        if img_fft_log[x][y]-prosek>razlika:
            img_fft_log[x][y] = prosek
            img_fft_log[img.shape[0]-x][img.shape[1]-y] = prosek
        for y in range(1, img.shape[1]-2):
            prosek = (img_fft_log[x-1][y-1] + img_fft_log[x-1][y] + img_fft_log[x-1][y+1] + img_fft_log[x][y-1] + img_fft_log[x][y+1] + img_fft_log[x+1][y-1] + img_fft_log[x+1][y] + img_fft_log[x+1][y+1])/8 #unutrasnji pikseli
            if img_fft_log[x][y]-prosek>razlika:
                img_fft_log[x][y] = prosek
                img_fft_log[img.shape[0]-x][img.shape[1]-y] = prosek
        y = img.shape[1]-1
        prosek = (img_fft_log[x-1][y] + img_fft_log[x-1][y-1] + img_fft_log[x][y-1] + img_fft_log[x+1][y] + img_fft_log[x+1][y-1])/5 #desna ivica
        if img_fft_log[x][y]-prosek>razlika:
            img_fft_log[x][y] = prosek
            img_fft_log[img.shape[0]-x][img.shape[1]-y] = prosek
    plt.imshow(img_fft_log)
    plt.show()
    img_filtered = inverse_fft(img_fft_log, img_pha)
    return img_filtered

def poboljsano3(img, razlika):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_pha = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    plt.imshow(img_fft_log)
    plt.show()

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            k = 0
            suma = 0
            # napravi kvadratnu masku oko piksela
            for i in range(-1, 2):
                for j in range(-1, 2):
                    yy = y + i
                    xx = x + j
                    if 0 <= yy < img.shape[1] and 0 <= xx < img.shape[0] and (xx != x or yy != y):
                        suma += img_fft_log[yy, xx]
                        k = k + 1

            prosek= suma// k
            if img_fft_log[x][y] - prosek > razlika:
                img_fft_log[x][y] = prosek
                img_fft_log[img.shape[0] - x-1][img.shape[1] - y-1] = prosek


    plt.imshow(img_fft_log)
    plt.show()
    img_filtered = inverse_fft(img_fft_log, img_pha)
    return img_filtered


if __name__ == '__main__':
    img = cv2.imread("./Materijal/slika_0.png") # OpenCV ucitava sliku u formatu BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # pretvaranje slike u nijanse sive (grayscale)

    plt.imshow(img, cmap='gray')
    plt.show()

    center = (256, 256) # slika je 512x512 pa ce centar biti na poziciji (256, 256)
    radius = 100 # radius idealnog low ili high pass filtra
    razlika = 5

    #plt.imshow(standardno(img), cmap='gray')
    #plt.imshow(poboljsano1(img, center,radius), cmap='gray')
    #plt.imshow(poboljsano2(img, razlika), cmap='gray')
    plt.imshow(poboljsano3(img, razlika), cmap='gray')
    plt.show()