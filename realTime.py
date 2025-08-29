import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

checkpoint_path = "checkpoints/fold1_best.pth"   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

idx_to_class = {0: "Bad Posture", 1: "Good Posture"}

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame (OpenCV BGR -> PIL RGB)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        label = idx_to_class[pred.item()]

    color = (0,255,0) if label == "Good Posture" else (0,0,255)
    cv2.putText(frame, label, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)

    cv2.imshow("Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
