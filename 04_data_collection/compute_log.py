import json
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client


decision_log = []
LOOP_TRACKS_NUMS = []
BEATS_PER_LOOP = 8
BASE_BPM = 120
decision_elements = []

def loopStart_handler(address, *args):
	global LOOP_TRACKS_NUMS, decision_log
	if args[0] == 1:
		print()
		print('Creating new decision log')
		print('-'*50)
		print()
		decision_log = []
		LOOP_TRACKS_NUMS = []
	elif args[0] == 0:
		print()
		print('-'*50)
		print('Saving decision log')
		with open(f'decisions_log.json', 'w', encoding='utf-8') as f:
			json.dump(decision_log, f, ensure_ascii=False, indent=4)


def hello_handler(address, *args):
	global LOOP_TRACKS_NUMS
	LOOP_TRACKS_NUMS.append(args[0])
	print(f'Loop tracks: {LOOP_TRACKS_NUMS}')

def reset_handler(address, *args):
	print('RESET')

def beatsPerLoop_handler(address, *args):
	global BEATS_PER_LOOP
	BEATS_PER_LOOP = args[0]
	print(f"BEATS_PER_LOOP: {BEATS_PER_LOOP}")

def baseBpm_handler(address, *args):
	global BASE_BPM
	BASE_BPM = args[0]
	print(f"BASE_BPM: {BASE_BPM}")

def endOfSubdivs_handler(address, *args):
	global decision_log, decision_elements
	print(f'End of subdivision: {args[0]}')
	new_log = {}
	new_log["subdivision_index (m)"] = args[0]
	new_log["decisions"] = []
	if not decision_elements:
		decision = {}
		decision["decision_type"] = "R"
		decision["loop_track (i)"] = None
		decision["num_beats (T_l)"] = None
		new_log["decisions"].append(decision)
		print(decision)
	else:
		new_log["decisions"] = decision_elements
		print(decision_elements)
	decision_log.append(new_log)
	decision_elements = []

def decision_handler(address, *args):
	global decision_elements
	loop_track_number = int(address.split('/')[-1])
	num_beats = args[0]
	#print(f'Decision: {LOOP_TRACKS_NUMS.index(loop_track_number)}, {num_beats}')
	decision = {}
	decision["decision_type"] = "A"
	decision["loop_track (i)"] = LOOP_TRACKS_NUMS.index(loop_track_number)
	decision["num_beats (T_l)"] = num_beats
	decision_elements.append(decision)


def default_handler(address, *args):
	print(f"DEFAULT {address}: {args}")


if __name__ == '__main__': 


	ip = "127.0.0.1" # localhost
	#port_snd = 6667 # send port to PD
	port_rcv = 6666 # receive port from PD

	## OSC SERVER
	# define dispatcher
	dispatcher = Dispatcher()
	dispatcher.map("/MOTHER_LOOPER", loopStart_handler)
	dispatcher.map("/HELLO", hello_handler)
	dispatcher.map("/RESET", reset_handler)
	dispatcher.map("/BEATS_PER_LOOP", beatsPerLoop_handler)
	dispatcher.map("/BASE_BPM", baseBpm_handler)
	dispatcher.map("/END_OF_SUBDIV", endOfSubdivs_handler)
	dispatcher.map("/decision/*", decision_handler)
	dispatcher.set_default_handler(default_handler)

	# define client
	#client = udp_client.SimpleUDPClient(ip, port_snd)

	# define server
	server = BlockingOSCUDPServer((ip, port_rcv), dispatcher)
	server.serve_forever()  # Blocks forever

