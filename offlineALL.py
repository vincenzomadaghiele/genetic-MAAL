import sys
import argparse
import warnings
from offlineALLclass import AutonomousLooperOffline


if __name__ == '__main__': 

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--SOUNDFILE_FILEPATH', type=str, default='./00_corpus/USE_CASE_1.wav',
						help='name of the folder containing the soundfile')
	parser.add_argument('--CONFIG_FILEPAHT', type=str, default='./config.json',
						help='path to the configuration file')
	parser.add_argument('--OUTPUT_DIR_PATH', type=str, default="./01_output_offline",
						help='path to the output file')
	parser.add_argument('--VERBOSE', type=int, default=1,
						help='text feedback 0=none, 1=decision log, 2=descriptor values')
	parser.add_argument('--IGNORE_WARNINGS', type=bool, default=False,
						help='ignore warnings')
	args = parser.parse_args(sys.argv[1:])

	## DEFINE SCRIPT PARAMETERS
	soundfile_filepath = args.SOUNDFILE_FILEPATH
	config_filepath = args.CONFIG_FILEPAHT
	output_dir_path = args.OUTPUT_DIR_PATH
	verbose = args.VERBOSE
	ignorewarnings = args.IGNORE_WARNINGS

	if ignorewarnings: 
		warnings.filterwarnings("ignore")

	looper = AutonomousLooperOffline(
									soundfile_filepath, 
									config_filepath=config_filepath, 
									plotFlag=False, 
									verbose=verbose
									)
	looper.computeLooperTrack(output_dir_path)
