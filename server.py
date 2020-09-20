from concurrent import futures
import grpc

from translation_service_pb2_grpc import TranslationServiceServicer, add_TranslationServiceServicer_to_server
from translation_service_pb2 import TranslatedReply

from PIL import Image

import cv2
import numpy as np

from yolo import YOLO
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.models import load_model

img_margin = 100
size = (848, 640)

yolo: YOLO = None
christina = None

_count = 0

chars = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
    26: 'nothing'
}

def load_all_models():
    global yolo, model
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])

    # Load A.L.I.C.E
    model = load_model('alicev3')


class TranslationService(TranslationServiceServicer):
    def Translate(self, request, context):
        global _count
        image = Image.frombuffer('RGBA', size, request.pixels, 'raw').convert('RGB')
        print('Got new request')
        mat = np.array(image)[:, :, ::-1].copy()  # Convert RGB to BGR
        width, height, inference_time, results = yolo.inference(mat)

        if len(results) > 0:
            detection = results[0]  # we only take one result
            # for detection in results:
            id, name, confidence, x, y, w, h = detection

            print("%s with %s confidence" % (name, round(confidence, 2)))
            cv2.imwrite("./export.jpg", mat)
            cv2.imwrite("./export_detected_ori.jpg", mat[y:y + h, x:x + w])

            detection_mat = mat[np.clip(y - img_margin, 0, size[1]):np.clip(y + h + img_margin, 0, size[1]),
                            np.clip(x - img_margin, 0, size[0]):np.clip(x + w + img_margin, 0, size[0])]
            detection_mat = cv2.rotate(detection_mat, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite("./export_detected.jpg",
                        detection_mat)



            # img_gray = cv2.cvtColor(detection_mat, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(detection_mat, (75, 75), interpolation=cv2.INTER_AREA)
            # img = img / 255
            # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ## FOR SAVING TRAINING DATA
            cv2.imwrite('./data2/'+str(_count)+'.jpg', img)
            _count+=1
            predicted_index = np.argmax(model.predict(img.reshape((-1, 75, 75, 3))))
            predicted_char = chars[predicted_index]

            print('Predicted Char: '+str(predicted_char))

            return TranslatedReply(char=predicted_char)
        new_mat = cv2.rotate(cv2.resize(mat, (75, 75)), cv2.ROTATE_90_COUNTERCLOCKWISE)
        predicted_index = np.argmax(model.predict(new_mat.reshape((-1, 75, 75, 3))))
        predicted_char = chars[predicted_index]
        print('Predicted Char: ' + str(predicted_char))
        return TranslatedReply(char=predicted_char)#'_'


def serve():
    load_all_models()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_TranslationServiceServicer_to_server(
        TranslationService(), server)
    server.add_insecure_port('192.168.1.1:5000')
    server.start()
    print('Started Server')
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
