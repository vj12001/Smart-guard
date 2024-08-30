import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import shutil

# Initialize FaceNet
embedder = FaceNet()
source_directory = "D:\\SSearch project\\Clash of Titans\\sample images folder"
def prepare_and_augment_data(source_directory, augmentations=10):
    """
    Organizes student images into folders based on their names and performs data augmentation.

    Parameters:
    - source_directory (str): Path to the directory containing student images.
    - augmentations (int): Number of augmented images to generate per original image.
    """
    if 'data_augmentation_done' not in st.session_state:
        # Create the target directory inside the source directory
        target_directory = os.path.join(source_directory, "organized_student_images")
        os.makedirs(target_directory, exist_ok=True)

        # Initialize the ImageDataGenerator with augmentation parameters
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Loop through each file in the source directory
        for filename in os.listdir(source_directory):
            if filename.endswith(".jpg"):
                # Extract the student name from the filename
                student_name = os.path.splitext(filename)[0]

                # Create a directory for each student
                student_directory = os.path.join(target_directory, student_name)
                os.makedirs(student_directory, exist_ok=True)

                # Copy the original image to the student's directory
                source_file = os.path.join(source_directory, filename)
                target_file = os.path.join(student_directory, filename)
                shutil.copy(source_file, target_file)
                print(f"Copied {filename} to {student_directory}")

                # Load the image for augmentation
                img = load_img(target_file)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)  # Reshape for the ImageDataGenerator

                # Generate and save augmented images
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=student_directory, save_prefix=student_name, save_format='jpg'):
                    i += 1
                    if i >= augmentations:  # Generate the specified number of augmented images per original image
                        break
                print(f"Generated {augmentations} augmented images for {student_name}")

        print("Data organization and augmentation completed.")
        # Mark data augmentation as done
        st.session_state.data_augmentation_done = True
    else:
        print("Data augmentation already completed.")

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detected_faces = self.detector.detect_faces(img)
        if detected_faces:
            x, y, w, h = detected_faces[0]['box']
            x, y = abs(x), abs(y)
            face = img[y:y+h, x:x+w]
            face_arr = cv2.resize(face, self.target_size)
            return face_arr
        return None

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            if im_name.endswith(".jpg"):
                path = os.path.join(dir, im_name)
                face = self.extract_face(path)
                if face is not None:
                    FACES.append(face)
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            if os.path.isdir(path):
                FACES = self.load_faces(path)
                labels = [sub_dir] * len(FACES)
                self.X.extend(FACES)
                self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    # 4D (Nonex160x160x3)
    yhat = embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

def prepare_and_load_faces(source_directory, augmentations=10):
    prepare_and_augment_data(source_directory, augmentations)
    faceloading = FACELOADING(os.path.join(source_directory, "organized_student_images"))
    X, Y_label = faceloading.load_classes()
    EMBEDDED_X = []
    count = 0
    for img in X:
        EMBEDDED_X.append(get_embedding(img))
        count += 1
        print(count)

    EMBEDDED_X = np.asarray(EMBEDDED_X)
    return EMBEDDED_X, Y_label


if 'embeddings_loaded' not in st.session_state:
    EMBEDDED_X, Y_label = prepare_and_load_faces(source_directory)
    st.session_state.EMBEDDED_X = EMBEDDED_X
    st.session_state.Y_label = Y_label
    st.session_state.embeddings_loaded = True
else:
    EMBEDDED_X = st.session_state.EMBEDDED_X
    Y_label = st.session_state.Y_label

haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
attendance_list = []
unknown_count = 0
known_unknowns = []
threshold_main = 0.01

# Load existing attendance file if it exists
attendance_file = "attendance.xlsx"
if os.path.exists(attendance_file):
    df_existing = pd.read_excel(attendance_file)
    existing_names = df_existing['Name'].tolist()
else:
    df_existing = pd.DataFrame(columns=["Name", "Time"])
    existing_names = []

def save_attendance(attendance_list):
    df_new = pd.DataFrame(attendance_list, columns=["Name", "Time"])
    df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset=['Name'], keep='first')
    df_combined.to_excel(attendance_file, index=False)

def save_unknown(frame, count):
    filename = f"D:\\SSearch project\\imposters\\Unidentified persons\\unknown_{count}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Unknown person saved as {filename}")

def distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def cosine_similarity(embedding, Embedded_X):
    norm_embedding = embedding / np.linalg.norm(embedding)
    norm_Embedded_X = Embedded_X / np.linalg.norm(Embedded_X, axis=1, keepdims=True)
    return np.dot(norm_Embedded_X, norm_embedding)

def recognize_face(embedding, Embedded_X, Y_label, threshold=0.7):
    similarities = cosine_similarity(embedding, Embedded_X)
    best_match_idx = np.argmax(similarities)
    if similarities[best_match_idx] >= threshold:
        return Y_label[best_match_idx]
    return "Unknown"

class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.unknown_count = 0

    def process_frame(self, frame):
        global unknown_count
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        recognized_names = []

        for x, y, w, h in faces:
            face_img = rgb_img[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))
            ypred = get_embedding(face_img)

            if ypred is None:
                continue

            final_name = recognize_face(ypred, EMBEDDED_X, Y_label)
            recognized_names.append(final_name)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
            cv2.putText(frame, str(final_name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for name in recognized_names:
            if name == "Unknown":
                if known_unknowns:
                    similarities = cosine_similarity(ypred, known_unknowns)
                    if np.max(similarities) > 0.85:
                        continue
                self.unknown_count += 1
                save_unknown(frame, self.unknown_count)
                known_unknowns.append(ypred)
            elif name not in existing_names:
                attendance_list.append([name, current_time])
                existing_names.append(name)

        save_attendance(attendance_list)
        
        return frame

    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        processed_frame = self.process_frame(frame)
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

def main():
    st.title("Real-Time Face Detection Application")
    activities = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by Divyesh Parmar and Raam    
            Email: divyeshparmar200@gmail.com
            [LinkedIn](https://www.linkedin.com)""")

    if choice == "Home":
        st.markdown("""
            <div style="background-color:#6D7B8D;padding:10px">
                <h4 style="color:white;text-align:center;">
                Face Detection and Recognition Application using OpenCV, FaceNet model, and Streamlit.
                </h4>
            </div>
            <br>
            """, unsafe_allow_html=True)
        
        st.write("""
            The application provides the following functionalities:

            1. **Real-time face detection using a webcam feed:** Detects faces in a live video stream using the webcam.
            
            2. **Real-time face identification:** Recognizes and identifies faces from the webcam feed against a known database of faces.
            
            3. **Attendance tracking:** Automatically records attendance by matching faces with the existing database and saves it to an Excel file.
            
            4. **Unknown face detection and saving:** Detects unknown faces and saves their images for later review.
            """)

    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect faces")
        prepare_and_augment_data(source_directory="D:\\SSearch project\\Clash of Titans\\sample images folder")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, 
                        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                        video_processor_factory=FaceDetectionProcessor)

    elif choice == "About":
        st.subheader("About this app")
        st.markdown("""
            <div style="background-color:#6D7B8D;padding:10px">
                <h4 style="color:white;text-align:center;">
                Real-time face detection application using OpenCV and Streamlit.
                </h4>
            </div>
            <br>
            """, unsafe_allow_html=True)
        st.write("""
            This application detects faces in real-time using a webcam feed.
            """)

if __name__ == "__main__":
    main()
