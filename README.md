# Smart-guard
Real-Time Imposter Detection
Objective 
The objective of this project is to develop a real-time imposter detection system that 
enhances security by accurately identifying and logging unknown attendees. This system 
allows users to upload photos of individuals they recognize or want to allow into a specific 
event or place. If an unknown individual, not present in the provided photos, attends the 
event or enters the place, the model will capture their face photo and save it along with a 
timestamp. Additionally, if a recognized individual is detected, their attendance is stored in 
an Excel file along with the time and their name. Leveraging advanced computer vision and 
machine learning techniques, this system provides a robust solution for monitoring and 
securing events and places, ensuring that only recognized individuals are allowed into 
designated areas. 
Key Features 
• Real-Time Detection: Continuous monitoring and detection of faces in real time. 
• Logging and Alerts: Captures and logs images of unknown individuals along with 
timestamps, enhancing security measures. 
• High Accuracy: Utilization of MTCNN and FaceNet ensures accurate face detection 
and embedding creation. 
• Efficient Classification: SVM classifier provides robust performance in 
distinguishing between known and unknown faces. 
Technologies and Libraries Used 
• OpenCV: For capturing video, image processing, and drawing bounding boxes 
around faces. 
• TensorFlow: Utilized for loading and using pre-trained models (FaceNet). 
• scikit-learn: Used for training and using the SVM classifier for face recognition. 
• Keras: - FaceNet: A pre-trained model for generating face embeddings. - ImageDataGenerator: For performing data augmentation on images. 
• Pandas: For handling and storing attendance data in Excel files. 
• MTCNN: A library for detecting faces in images. 
• Pickle: For saving and loading the trained SVM model. 
• OS: For handling file operations and environment configurations. 
• DateTime: For handling date and time operations.
