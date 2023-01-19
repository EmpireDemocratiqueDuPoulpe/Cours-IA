from pathlib import Path
import cv2
import numpy

def show_img(img, title: str = "Image") -> None:
    cv2.imshow(title, img)
    cv2.waitKey(0)


# ### 1 - Line detection ###############################################################################################
def detect_lines(image, horizontal: bool = True, vertical: bool = True) -> None:
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_thresh = cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image_result = image_grayscale.copy()
    image_result = cv2.cvtColor(image_result, cv2.COLOR_GRAY2RGB)

    # Detect horizontal lines
    if horizontal:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        contours = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for c in contours:
            cv2.drawContours(image_result, [c], -1, (36, 255, 12), 2)

    # Detect vertical lines
    if vertical:
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        detect_vertical = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        contours = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for c in contours:
            cv2.drawContours(image_result, [c], -1, (36, 12, 255), 2)

    show_img(image_result, title="Line detection")


# ### 2 - Image translation ############################################################################################
def translate(image, x: int, y: int) -> None:
    height, width = image.shape[:2]
    translation_matrix = numpy.array([
        [ 1, 0, x ],
        [ 0, 1, y ]
    ], dtype=numpy.float32)

    image_translated = cv2.warpAffine(src=image, M=translation_matrix, dsize=(height, width))
    show_img(image_translated, title="Translation")


# ### 3 - Blur #########################################################################################################
def blur(image, k: tuple[int, int]) -> None:
    image_blurred = cv2.blur(image, k)
    show_img(image_blurred, title="Blur")


# ### 4 - Gaussian blur ################################################################################################
def gaussian_blur(image, k: tuple[int, int]) -> None:
    image_blurred = cv2.GaussianBlur(image, k, 0)
    show_img(image_blurred, title="Gaussian blur")


# ### Main #############################################################################################################
def main():
    # Load the image
    image = cv2.imread(str((Path(__file__).resolve().parent / "data" / "bedrock.png")))

    # 1. Lines detection
    detect_lines(image=image)

    # 2. Image translation
    translate(image=image, x=100, y=100)

    # 3 - Blur
    blur(image=image, k=(10, 10))

    # 4 - Gaussian blur
    gaussian_blur(image=image, k=(15, 15))


if __name__ == '__main__':
    main()
