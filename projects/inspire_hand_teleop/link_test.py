#!/usr/bin/env python3
"""Tiny TCP reachability test for the study link (default port 5070).

  Laptop (listener):    python3 link_test.py listen [port]
  IT PC  (connector):   python3 link_test.py connect <laptop-ip> [port]

It mirrors the real link (the listener = the machine with the reachable IP). If the
connector prints CONNECTED, put that <laptop-ip> in the IT PC's Setup-bar "Link IP".
Python 3.6+; no arguments to paste, so no copy-paste indentation issues.
"""
import socket
import sys


def listen(port):
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)   # accept IPv4 too
    except OSError:
        pass
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("::", port))
    s.listen(1)
    print("LISTENING on [::]:%d (IPv4+IPv6) -- waiting for the other machine ..." % port)
    c, a = s.accept()
    print("GOT CONNECTION FROM", a)
    print("recv:", c.recv(200))
    c.sendall(b"hello-from-listener\n")
    c.close()
    s.close()
    print("OK -- the link path works.")


def connect(host, port):
    print("connecting to %s:%d ..." % (host, port))
    s = socket.create_connection((host, port), timeout=5)
    s.sendall(b"hi-from-connector")
    print("CONNECTED -- reply:", s.recv(100))
    s.close()
    print("OK -- put  %s  in this machine's Setup-bar 'Link IP'." % host)


if __name__ == "__main__":
    args = sys.argv
    if len(args) >= 2 and args[1] == "listen":
        listen(int(args[2]) if len(args) > 2 else 5070)
    elif len(args) >= 3 and args[1] == "connect":
        connect(args[2], int(args[3]) if len(args) > 3 else 5070)
    else:
        print("usage: python3 link_test.py listen [port]")
        print("       python3 link_test.py connect <laptop-ip> [port]")
        sys.exit(1)
