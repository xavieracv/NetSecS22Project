# Testing Setup

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
import pyshark
import tensorflow as tf

#SERVER_IP = '18.213.140.149'
#SERVER_PORT = 50007

SERVER_IP   = '1.1.1.1'
SERVER_PORT = 53
BUFFER_SIZE = 1024

sus_port      = 0
sus_ip_src    =  ""
sus_ip_dest   =  ""

def analyze(model, pkt):

    try:

        queryname =  pkt['dns'].qry_name
        respname  =  pkt['dns'].cname
    except:
        return

    if (queryname == "cmVxdWVzdGluZyBpbnN0cnVjdGlvbnM0"):
        print("WOOOOOOOOOO")

    print(queryname) # will look like cmVxdWVzdGluZyBpbnN0cnVjdGlvbnM0.netsec-project.me when malicious
    print(respname)  # will look like RERPUyAxMjguMTQzLjIyLjExOQ00.netsec-project.me when malicious

    # Run domains through Model...
    # Alert user if positive classification


def detect():
    model = 3
    #model = tf.keras.models.load_model("dns_model.h5")
    capture = pyshark.LiveCapture(interface='en0', bpf_filter='udp', display_filter='dns')

    for packet in capture.sniff_continuously():
        analyze(model, packet)

def main():
    # while(1):
    detect()
    #time.sleep(5)


if __name__=="__main__":
    main()
