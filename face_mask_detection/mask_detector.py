import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 100  # Kích thước ảnh đầu vào

# 1️ Load và tiền xử lý dữ liệu từ thư mục dataset
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "D:\\HocTap\\Trí tuệ nhân tạo\\face_mask_detection\\dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    "D:\\HocTap\\Trí tuệ nhân tạo\\face_mask_detection\\dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 2️ Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Lớp đầu ra (0: Mask, 1: No Mask)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3️ Huấn luyện mô hình
print("\nĐang huấn luyện mô hình...")
model.fit(train_data, validation_data=val_data, epochs=10)  # Giới hạn 5 epoch

# 4️ Lưu mô hình đã huấn luyện
model.save("mask_detector.h5")
print("\nMô hình đã được lưu thành công!")

# 5️ Nhận diện khẩu trang bằng webcam
print("\nĐang mở camera...")
model = tf.keras.models.load_model("mask_detector.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)  # Mở webcam

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = np.expand_dims(face, axis=0) / 255.0  # Chuẩn hóa ảnh

        prediction = model.predict(face)[0][0]
        
        # Sửa lại logic màu
        if prediction < 0.5:
            label = "No Mask"
            color = (0, 0, 255)  # Đỏ khi KHÔNG đeo khẩu trang
        else:
            label = "Mask"
            color = (0, 255, 0)  # Xanh khi CÓ đeo khẩu trang

        # Vẽ khung và hiển thị nhãn
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Hiển thị cửa sổ nhận diện khuôn mặt
    cv2.imshow("Face Mask Detector", frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
