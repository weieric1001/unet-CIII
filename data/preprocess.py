from pathlib import Path
import cv2

from mtcnn import MTCNN
from tqdm import tqdm


def detect_face_MTCNN(image):
    detector = MTCNN()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)
    for result in results:
        bounding_box = result["box"]
        keypoints = result["keypoints"]
    return bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]


def cut_image(image, x, y, w, h):
    image_h, image_w = image.shape[:2]
    length = min(image_w, image_h)
    if image_w > image_h:
        x = max(0, x - (length - w) // 2)
        w = length
        return image[:, x : x + w]
    else:
        y = max(0, y - (length - h) // 2)
        h = length
        return image[y : y + h, :]


if __name__ == "__main__":
    folder = Path("C:/Users/weiwe/Documents/GitHub/unet-CIII/data/pre/front")
    files = list(folder.glob("*.jpg"))
    files.sort()

    save_folder = folder / "processed"
    save_folder.mkdir(exist_ok=True, parents=True)

    for image_path in tqdm(files):
        try:
            image = cv2.imread(str(image_path))
            min_len = min(image.shape[:2])
            x, y, w, h = detect_face_MTCNN(image)
            image = cut_image(image, x, y, w, h)
            image_name = image_path.name.split("-")[0]
            save_path = save_folder / f"{image_name}.jpg"
            cv2.imwrite(str(save_path), image)
        except:
            with open("error.txt", "a") as f:
                f.write(str(image_path) + "\n")
                f.close()
