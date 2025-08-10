from SpeechRecongnition import *
from Server import *
import threading
import time

HOST = "0.0.0.0"
PORT = 5001
clients = set()


class HandleCallback(Callback):
    def __init__(self, server):
        super().__init__()
        self.server = server

    def on_event(self, result):
        print(self.server.clients)
        sentence = result.get_sentence()
        print(sentence["text"])
        for client in list(self.server.clients):
            try:
                client.sendall(bytes(sentence["text"][-20:] + "\n", "utf-8"))
                # client.sendall(b"Group2\n")
            except:
                clients.remove(client)


class VoiceRecongnition(SpeechRecognition):
    def __init__(self):
        self.server = Server(HOST, PORT)

        callback = HandleCallback(self.server)
        t = threading.Thread(target=self.server.run, daemon=True)
        t.start()
        super().__init__(callback)


if __name__ == "__main__":
    vr = VoiceRecongnition()
    vr.start()