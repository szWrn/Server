import socket
from SpeechRecongnition import *
import threading

HOST = "0.0.0.0"
PORT = 5001
clients = set()


class HandleCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_event(self, result):
        sentence = result.get_sentence()
        print(sentence["text"])
        for client in list(clients):
            try:
                client.sendall(sentence["text"])
            except:
                clients.remove(client)


class Server:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((HOST, PORT))
        self.server.listen(5)

        self.speechrecognition = SpeechRecognition(HandleCallback())

    def run(self):
        threading.Thread(target=self.speechrecognition.start, daemon=True)

        while True:
            client_socket, addr = self.server.accept()
            clients.add(client_socket)


if __name__ == "__main__":
    server = Server()
    server.run()