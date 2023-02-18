import dlib
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import os
import tempfile

detector = dlib.get_frontal_face_detector()


def save(img, name, bbox, width=180, height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x:w]
    imgCrop = cv2.resize(imgCrop, (width, height))
    cv2.imwrite(name + '.jpg', imgCrop)


def crop_faces(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file.flush()
        temp_file.seek(0)
        image = cv2.imread(temp_file.name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        crops = []
        fit = 50
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
            x1 = max(0, x - fit)
            y1 = max(0, y - fit)
            x2 = min(image.shape[1], w + fit)
            y2 = min(image.shape[0], h + fit)
            crop = image[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (180, 227))
            crops.append(crop)
    return crops


def create_thumbnail_grid(filenames, folder_path, grid_size=(4, 4), thumb_size=(180, 227), margin=(10, 20)):
    margin_top_bottom, margin_left_right = margin
    thumb_width, thumb_height = thumb_size
    grid_rows, grid_cols = grid_size

    canvas_width = (thumb_width + margin_left_right * 2) * grid_cols
    canvas_height = (thumb_height + margin_top_bottom * 2) * grid_rows

    canvas = Image.new('RGB', (canvas_width, canvas_height),
                       (255, 255, 255, 255))

    for i, filename in enumerate(filenames):
        row = i // grid_cols
        col = i % grid_cols
        left = margin_left_right + col * (thumb_width + margin_left_right)
        top = margin_top_bottom + row * (thumb_height + margin_top_bottom)
        img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
        img = img.resize(thumb_size, Image.ANTIALIAS)
        canvas.paste(img, (left, top))

    return np.array(canvas)


def main():
    st.set_page_config(page_title="Crop People Out of a Photo",
                       page_icon=":guardsman:",
                       layout="wide")
    st.title("Crop People Out of a Photo")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        crops = crop_faces(uploaded_file)
        if len(crops) == 0:
            st.warning("No faces detected in the uploaded image.")
        else:
            st.write(f"Detected {len(crops)} faces.")
            cols = st.columns(4)
            for i, crop in enumerate(crops):
                with cols[i % 4]:
                    st.image(crop, use_column_width=True,
                             caption=f"Face {i+1}")


if __name__ == '__main__':
    main()
