from os import path
from sys import stderr
import argparse

import dpkt
from dpkt.utils import mac_to_str, inet_to_str, make_dict
from pprint import pprint

# import numpy as np
# import keras
# TODO: from model file, import pre-processing function


def verify_pcap(pcap_file):
    """
    Error checking on pcap file
    """
    if not path.isfile(pcap_file):
        print("ERROR: Please provide a valid file.", file=stderr)
        exit(1)
    if not pcap_file.endswith('.pcap'):
        print("ERROR: Please provide a pcap file with extension '.pcap'", file=stderr)
        exit(1)
    if path.getsize(pcap_file) == 0:
        print("ERROR: empty file:", pcap_file, file=stderr)
        exit(1)

def model_load():
    """
    Loads model from file
    """
    modelName = 'DGAModel'
    # TODO: place DGAModel in same directory and uncomment the following line
    # model = keras.models.load_model(modelName)
    model = None  # comment/remove me once prev is uncommented!
    print("Info for ", modelName)

    # ---------------------------------
    # GET ACCURACY
    # eval_generator # generates data for evaluating model
    # test_loss, test_acc = model.evaluate(eval_generator, verbose=2)
    # print(test_acc)
    return model

def print_packet(buf):
    """
    Print out information about each packet in a pcap
    Credit to: dpkt DNS example
    https://github.com/kbandla/dpkt/blob/master/examples/print_dns_truncated.py

    Args:
        buf: buffer of bytes for this packet

    Sample output:
        <class 'bytes'>
        Ethernet Frame:  00:1e:8c:ea:1a:b4 a4:2b:8c:f6:eb:81 2048
        IP: 10.0.2.191 -> 192.122.184.88   (len=52 ttl=64 DF=1 MF=0 offset=0)
    Sample DNS Query and Response:
        <class 'bytes'>
        Ethernet Frame:  00:1e:8c:ea:1a:b4 a4:2b:8c:f6:eb:81 2048
        IP: 10.0.2.191 -> 10.0.2.1   (len=61 ttl=255 DF=0 MF=0 offset=0)
        UDP: sport=49515 dport=53 sum=10225 ulen=41
        Queries: 1
            apis.google.com Type:1                                                                                                                                     
        Answers: 0
        ...
        <class 'bytes'>
        Ethernet Frame:  a4:2b:8c:f6:eb:81 00:1e:8c:ea:1a:b4 2048
        IP: 10.0.2.1 -> 10.0.2.191   (len=394 ttl=64 DF=0 MF=0 offset=0)
        UDP: sport=53 dport=49515 sum=34787 ulen=374
        Queries: 1
            apis.google.com Type:1
        Answers: 12
            apis.google.com: type: CNAME Answer: plus.l.google.com
            plus.l.google.com: type: A Answer: 74.125.225.38
            plus.l.google.com: type: A Answer: 74.125.225.39
            plus.l.google.com: type: A Answer: 74.125.225.40
            plus.l.google.com: type: A Answer: 74.125.225.41
            plus.l.google.com: type: A Answer: 74.125.225.46
            plus.l.google.com: type: A Answer: 74.125.225.32
            plus.l.google.com: type: A Answer: 74.125.225.33
            plus.l.google.com: type: A Answer: 74.125.225.34
            plus.l.google.com: type: A Answer: 74.125.225.35
            plus.l.google.com: type: A Answer: 74.125.225.36
            plus.l.google.com: type: A Answer: 74.125.225.37
    """
    print(type(buf))

    # Unpack the Ethernet frame (mac src/dst, ethertype)
    eth = dpkt.ethernet.Ethernet(buf)
    print('Ethernet Frame: ', mac_to_str(eth.src), mac_to_str(eth.dst), eth.type)

    # Make sure the Ethernet data contains an IP packet
    if not isinstance(eth.data, dpkt.ip.IP):
        print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
        return

    # Now unpack the data within the Ethernet frame (the IP packet)
    # Pulling out src, dst, length, fragment info, TTL, and Protocol
    ip = eth.data

    # Pull out fragment information (flags and offset all packed into off field, so use bitmasks)
    do_not_fragment = bool(ip.off & dpkt.ip.IP_DF)
    more_fragments = bool(ip.off & dpkt.ip.IP_MF)
    fragment_offset = ip.off & dpkt.ip.IP_OFFMASK

    # Print out the info
    print('IP: %s -> %s   (len=%d ttl=%d DF=%d MF=%d offset=%d)' %
          (inet_to_str(ip.src), inet_to_str(ip.dst), ip.len, ip.ttl, do_not_fragment,
           more_fragments, fragment_offset))

    # Check for UDP in the transport layer
    if isinstance(ip.data, dpkt.udp.UDP):

        # Set the UDP data
        udp = ip.data
        print('UDP: sport={:d} dport={:d} sum={:d} ulen={:d}'.format(udp.sport, udp.dport,
                                                                     udp.sum, udp.ulen))

        # Now see if we can parse the contents of the truncated DNS request
        try:
            dns = dpkt.dns.DNS()
            dns.unpack(udp.data)
        except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, Exception) as e:
            print('\nError Parsing DNS, Might be a truncated packet...')
            print('Exception: {!r}'.format(e))

    # Print out the DNS info
    print('Queries: {:d}'.format(len(dns.qd)))
    for query in dns.qd:
        print('\t {:s} Type:{:d}'.format(query.name, query.type))
    print('Answers: {:d}'.format(len(dns.an)))
    for answer in dns.an:
        if answer.type == 5:
            print('\t {:s}: type: CNAME Answer: {:s}'.format(answer.name, answer.cname))
        elif answer.type == 1:
            print('\t {:s}: type: A Answer: {:s}'.format(answer.name, inet_to_str(answer.ip)))
        else:
            pprint(make_dict(answer))

def detectDNSTunneling(pcap_file, model):
    """
    Runs DNS domains through pretrained model for
    DNS tunneling detection and flags suspicious packets
    """
    with open(pcap_file, 'rb') as f:
        pcap = None
        try:
            pcap = dpkt.pcap.Reader(f)
        except ValueError as v:
            print("ERROR: Could not read pcap file:", pcap_file, file=stderr)
            print(v)
            exit(1)
        
        print("Processing packet capture...")
        print("The following packets may be using DNS tunneling:")
        suspicious_count = 0
        for timestamp, buf in pcap:
            try:
                print_packet(buf)  # sample output
                # TODO: read packet as DNS query

                # TODO: extract domain from packet
                # domain = 

                # TODO: send domain through pre-processing pipeline 
                # (removes TLD, converts to ASCII int representations, pads, etc.)
                # processed_domain = process(domain)

                # Send processed domain into model.predict()
                # prediction = model.predict(processed_domain)

                # If model predicts that this domain is using DNS tunneling,
                # report packet number and domain name
                # if prediction == 1:
                #   suspicious_count += 1
                #   print(packet_num, domain)  # TODO: entire packet description?
                continue  # remove me eventually
            except Exception:
                # Silently ignore non-DNS packets
                continue
        print("Found", suspicious_count, "packets that may be using DNS tunneling")

def main():
    parser = argparse.ArgumentParser(description='Performs DNS tunneling detection on a packet capture')
    parser.add_argument('-f', '--pcap_file', required=True, type=str, help='Path to the .pcap file')
    args = parser.parse_args()

    pcap_file = args.pcap_file
    verify_pcap(pcap_file)

    dns_tunneling_detector = model_load()  # load the trained model
    detectDNSTunneling(pcap_file, dns_tunneling_detector)

if __name__=="__main__":
    main()
