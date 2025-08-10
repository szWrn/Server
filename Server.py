import socket

class Server:
    def __init__(self, host, port):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(5)

        self.clients = set()

    def run(self):
        # tm = threading.Thread(target=self.speechrecognition.start, daemon=True)
        # # tm.start()

        while True:
            print("new client")
            client_socket, addr = self.server.accept()
            self.clients.add(client_socket)
