# Loads our first model iteration and detects live traffic for malicious dns activity

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
from os import path
from sys import stderr
import argparse

import dpkt
from dpkt.utils import mac_to_str, inet_to_str, make_dict
from pprint import pprint

import pandas as pd
import keras
from dns_classifier import preprocess
from rm_tld import remove_tld
import numpy as np

#SERVER_IP = '18.213.140.149'
#SERVER_PORT = 50007

SERVER_IP   = '1.1.1.1'
SERVER_PORT = 53
BUFFER_SIZE = 1024

sus_port      = 0
sus_ip_src    =  ""
sus_ip_dst   =  ""

def model_load(model_file):
    """
    Loads model from file
    """
    model = keras.models.load_model(model_file)
    #print("Info for ", model_file)

    # ---------------------------------
    # GET ACCURACY
    # eval_generator # generates data for evaluating model
    # test_loss, test_acc = model.evaluate(eval_generator, verbose=2)
    # print(test_acc)
    #print(model.summary())

    return model



def analyze(model, pkt):
    global sus_ip_src, sus_ip_dst

    queryname = ""
    respname  = ""

    try:
        queryname =  pkt['dns'].qry_name
        respname  =  pkt['dns'].cname
    except:
        if (queryname == ""):
            return

    domain = []

    q_bool = False
    r_bool = False

    if (queryname != ""):
        domain.append(queryname)
        q_bool=True

    if (respname != ""):
        domain.append(respname)
        #r_bool=True

    #print(queryname) # will look like cmVxdWVzdGluZyBpbnN0cnVjdGlvbnM0.netsec-project.me when malicious
    #print(respname)  # will look like RERPUyAxMjguMTQzLjIyLjExOQ00.netsec-project.me when malicious


    # Run domains through Model...
    # Alert user if positive classification

    df = pd.DataFrame(domain, columns=['domain'])
    processed_domains, _ = preprocess(df, 253)  # to match maxLen the model was trained with
    predictions = model.predict(processed_domains)

    if (len(predictions) > 1):
        if (predictions[0].argmax() != 0):
            print("Suspicious DNS Activity:\n\t" + str(sus_ip_src) + "->" + str(sus_ip_dst) + "\n\t'" + queryname + "'")
        if (predictions[1].argmax() != 0):
            print("\t'" + respname  + "'")


    elif (len(predictions) == 1):
        if (predictions[0].argmax() != 0):
            print("Suspicious DNS Activity:\n\t" + str(sus_ip_src) + "->" + str(sus_ip_dst) + "\n\t'" + queryname +"'")  if (q_bool) else print("Suspicious DNS Activity:\n\t" + str(sus_ip_src) + "->" + str(sus_ip_dst) + "\n\t'" + respname +"'")




def detect():
    global sus_ip_src, sus_ip_dst

    model = model = model_load("/Users/mattwalters/Documents/Sem2/dns_model.h5")

    #model = tf.keras.models.load_model("dns_model.h5")
    capture = pyshark.LiveCapture(interface='en0', bpf_filter='udp', display_filter='dns')

    for packet in capture.sniff_continuously():
        if ('IP' in packet):
            #print("hello")
            sus_ip_src    =  packet['ip'].src
            sus_ip_dst   =  packet['ip'].dst

            analyze(model, packet)




def main():
    # while(1):
    detect()
    #time.sleep(5)


if __name__=="__main__":
    main()
