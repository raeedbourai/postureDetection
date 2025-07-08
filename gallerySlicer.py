import cv2
import os

def sliceGallery(path, output, rows, cols, count):
    image = cv2.imread(path)
    if image is not None:
        height, width, _ = image.shape
        cell_h = height // rows
        cell_w = width // cols

        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * cell_w, r * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                crop = image[y1:y2, x1:x2]
                filename = f"person_{count}.jpg"
                cv2.imwrite(os.path.join(output, filename), crop)
                count += 1
    return count

count = 1

for folder in os.listdir('zoomGalleries'):
    for filename in os.listdir('zoomGalleries/' + folder):
        filename = os.path.join('zoomGalleries', folder) + '/' + filename
        print(filename, ' ', folder[0], ' ', folder[2])
        count = sliceGallery(filename, 'cropped_people/', int(folder[0]), int(folder[2]), count)
