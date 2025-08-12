from Server import *
import threading

dir = {"1": "<|------>", "2": "<--|---->", "3": "<---|--->", "4": "<----|-->", "5": "<------|>", "6": "<------->"}

if __name__ == "__main__":
    server = Server("0.0.0.0", 5001)
    st = threading.Thread(target=server.run, daemon=True)
    st.start()
    while (True):
        a = input()
        for client in list(server.clients):
            try:
                client.sendall(bytes(dir[a] + "\n", "utf-8"))
                # client.sendall(b"Group2\n")
            except:
                server.clients.remove(client)
        
        