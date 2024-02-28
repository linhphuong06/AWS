from fastapi import FastAPI, UploadFile
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import io
import tensorflow
import numpy as np
import cv2 as cv


app = FastAPI()

# class CustomMobileNet(nn.Module):
#     def __init__(self):
#         super(CustomMobileNet, self).__init__()
#         # Định nghĩa kiến trúc mô hình của bạn

#     def forward(self, x):
#         # Quá trình lan truyền thuận của mô hình
#         pass

# # Khởi tạo mô hình CustomMobileNet
# model = CustomMobileNet()
model = tensorflow.keras.models.load_model("./model.h5")

# Endpoint dự đoán
@app.post('/predict')
async def predict(file: UploadFile):
    # Đọc dữ liệu từ tệp tin
    contents = await file.read()

    # Tiền xử lý dữ liệu đầu vào (tùy thuộc vào mô hình của bạn)
    # transform = transforms.Compose([transforms.Resize((224, 224)),  # Thay đổi kích thước ảnh theo yêu cầu
    #                                 transforms.ToTensor(),  # Chuyển đổi ảnh thành tensor
    #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                      std=[0.229, 0.224, 0.225])])  # Chuẩn hóa ảnh

    # Chuyển đổi dữ liệu ảnh từ bytes sang PIL Image
    image = Image.open(io.BytesIO(contents))
    image = np.array(image)
    print(type(image))
    # image = cv.imread(contents)
    label_name = ['Inocybe', 'Agaricus', 'Entoloma', 'Pluteus', 'Suillus', 'Boletus', 'Amanita', 'Exidia', 'Cortinarius', 'Russula', 'Hygrocybe', 'Lactarius']
    # Áp dụng các biến đổi
    image = cv.resize(image,(224,336))
    img = np.array([image])
    pred = model.predict(img)
    pred_labels = np.argmax(pred, axis=1)
    output_name = label_name[pred_labels[0]]  # Thêm chiều batch (unsqueeze)

    # Thực hiện dự đoán bằng mô hình PyTorch
    # with torch.no_grad():
    #     prediction = model(input_tensor)

    # Xử lý kết quả đầu ra (tùy thuộc vào mô hình của bạn)
    # ...

    # Trả về kết quả dự đoán
    print(output_name)
    return {pred_labels[0],output_name}
