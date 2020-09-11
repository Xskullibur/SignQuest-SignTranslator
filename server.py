
from concurrent import futures
import grpc

from translation_service_pb2_grpc import TranslationServiceServicer, add_TranslationServiceServicer_to_server
from translation_service_pb2 import TranslatedReply

from PIL import Image


class TranslationService(TranslationServiceServicer):
    def Translate(self, request, context):
        image = Image.frombuffer('RGBA', (28, 28), request.pixels, 'raw').convert('RGB')
        print('Got request from {}'.format(request.pixels))
        image.save('./test.jpg')
        return TranslatedReply(char='a')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_TranslationServiceServicer_to_server(
        TranslationService(), server)
    server.add_insecure_port('192.168.1.1:5000')
    server.start()
    print('Started Server')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()