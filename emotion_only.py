import os
import glob
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from ResidualMaskingNetwork.models import densenet121, resmasking_dropout1
import pickle

def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image

def main():
    # Face detection stuff:
    net = cv2.dnn.readNetFromCaffe("ResidualMaskingNetwork/deploy.prototxt.txt", "ResidualMaskingNetwork/res10_300x300_ssd_iter_140000.caffemodel")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    FER_2013_EMO_DICT = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    # load configs and set random seed
    configs = json.load(open('ResidualMaskingNetwork/configs/fer2013_config.json'))
    image_size = (configs['image_size'], configs['image_size'])
    # Prediction stuff
    model = resmasking_dropout1(in_channels=3, num_classes=7)
    state = torch.load('ResidualMaskingNetwork/saved/checkpoints/Z_resmasking_dropout1_rot30_2019Nov30_13.32', map_location=torch.device('cpu') )
    model.load_state_dict(state['net'])
    model.eval()

    vid = cv2.VideoCapture(0)

    # cv2.namedWindow('disp')
    # cv2.resizeWindow('disp', width=800)

    with torch.no_grad():
        last_3 = np.zeros((1,7))
        count = 0
        while True:
            count += 1
            ret, frame = vid.read()
            if frame is None or ret is not True:
                continue
            try:
                frame = np.fliplr(frame).astype(np.uint8)
                # frame += 50
                h, w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # gray = frame

                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                faces = net.forward()

                for i  in range(0, faces.shape[2]):
                    confidence = faces[0, 0, i, 2]
                    if confidence < 0.5:
                        continue
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    start_x, start_y, end_x, end_y = box.astype("int")

                    #covnert to square images
                    center_x, center_y = (start_x + end_x) // 2 , (start_y + end_y) // 2
                    square_length = ((end_x - start_x ) + (end_y - start_y)) // 2 // 2

                    square_length *= 1.1

                    start_x = int(center_x - square_length)
                    start_y = int(center_y - square_length)
                    end_x = int(center_x + square_length)
                    end_y = int(center_y + square_length)


                    cv2.rectangle(frame , (start_x, start_y), (end_x, end_y), (179, 255, 179), 2)
                    # cv2.rectangle(frame , (x, y), (x + w, y + h), (179, 255, 179), 2)

                    # face = gray[y:y + h, x:x + w]
                    face = gray[start_y:end_y, start_x:end_x]

                    face = ensure_color(face)

                    face = cv2.resize(face, image_size)
                    face = transform(face)
                    face = torch.unsqueeze(face, dim=0)

                    output = torch.squeeze(model(face), 0)
                    proba = torch.softmax(output, 0)
                    last_3 += proba.numpy()

                    # emo_idx = torch.argmax(proba, dim=0).item()
                    emo_proba, emo_idx = torch.max(proba, dim=0)
                    emo_idx = emo_idx.item()
                    emo_proba = emo_proba.item()

                    emo_label = FER_2013_EMO_DICT[emo_idx]

                    label_size, base_line = cv2.getTextSize('{}: 000'.format(emo_label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                    # cv2.rectangle(
                    #     frame,
                    #     (end_x, start_y + 1 - label_size[1]),
                    #     (end_x + label_size[0], start_y + 1 + base_line),
                    #     (223, 128, 255),
                    #     cv2.FILLED
                    # )
                    # cv2.putText(
                    #     frame,
                    #     '{} {}'.format(emo_label, int(emo_proba * 100)),
                    #     (end_x, start_y + 1),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.8, (0, 0, 0), 2
                    # )
                    cv2.imshow('disp', frame)

                    dbfile = open('probs.p', 'wb')
                    pickle.dump(proba.numpy(), dbfile)
                    print("Wrote to file")

                if cv2.waitKey(1) == ord('q'):
                    break

            except:
                continue
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
