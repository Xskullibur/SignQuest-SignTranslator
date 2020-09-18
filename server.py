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

def load_all_models():
    global yolo, model
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])

    # Load Christina
    model = load_model('christinav8')


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

            ## FOR SAVING TRAINING DATA
            cv2.imwrite('./data2/'+str(_count)+'.jpg', detection_mat)
            _count+=1

            img_gray = cv2.cvtColor(detection_mat, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            img = img / 255
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            predicted_index = np.argmax(model.predict(img.reshape((-1, 28, 28, 1))))
            predicted_char = chr(ord('A') + predicted_index)

            print('Predicted Char: '+str(predicted_char))

            return TranslatedReply(char=predicted_char)
        return TranslatedReply(char='_')


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
