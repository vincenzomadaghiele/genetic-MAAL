import json
import os
import threading
import collections
import librosa
import numpy as np
import numpy.lib.recfunctions
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client
import matplotlib.pyplot as plt


class LoopAgent():
	def __init__(self, 
			loop_track_num=0,
			sr=44100,
			fft_window=1024,
			fft_hopSize=512,
			BEATS_PER_LOOP=8,
			MIN_BEATS_PER_LOOP=4,
			BASE_BPM=120,
			RHYTHM_SUBDIVISIONS=16,
			MIN_REPETITIONS=4,
			MAX_REPETITIONS=32
		):

		# properties
		self.BEATS_PER_LOOP = int(BEATS_PER_LOOP)
		self.BASE_BPM = BASE_BPM
		self.RHYTHM_SUBDIVISIONS = int(RHYTHM_SUBDIVISIONS)
		self.MIN_BEATS_PER_LOOP = int(MIN_BEATS_PER_LOOP)
		self.MIN_REPETITIONS = int(MIN_REPETITIONS)
		self.MAX_REPETITIONS = int(MAX_REPETITIONS)
		self.verbose = 1
		self.loop_track_num = loop_track_num

		# LOAD LOOPING RULES
		self.looping_rules = []
		with open(f"rules-{self.loop_track_num}.txt") as f:
			lines = f.readlines()
		for line in lines:
			if line.split(';')[0] == "RULE":
				newrule = {}
			elif line.split(';')[0].split(":")[0] == "rule-name":
				newrule["rule-name"] = line.split(';')[0].split(":")[1]
			elif line.split(';')[0].split(":")[0] == "rule-type":
				newrule["rule-type"] = line.split(';')[0].split(":")[1]
			elif line.split(';')[0].split(":")[0] == "rule-threshold":
				newrule["rule-threshold"] = float(line.split(';')[0].split(":")[1])
				self.looping_rules.append(newrule)
		print(self.looping_rules)

		# BUFFER SIZES CONFIGURATIONS
		self.sr = sr
		self.FFT_WINDOW = fft_window
		self.FFT_HOP_SIZE = fft_hopSize
		self.N_BAR_SAMPLES = int(1 / (self.BASE_BPM / 60) * self.sr * self.BEATS_PER_LOOP) # number of samples in a bar: 1 / BPS * framerate * beats_per_bar
		self.N_FFT_FRAMES = int(self.N_BAR_SAMPLES / self.FFT_HOP_SIZE) + 1
		self.frames_per_beat = int(self.N_FFT_FRAMES / self.BEATS_PER_LOOP)

		# DEFINE ALL NAMES OF LOOPING CRITERIA
		# the rules in the list should be in the same order as the vector that computes them		
		self.RULE_NAMES = ["Harmonic similarity", "Harmonic movement - C", "Harmonic movement - D",
							"Melodic similarity", "Melodic trajectory - C", "Melodic trajectory - D",
							"Dynamic similarity", "Dynamic changes - C", "Dynamic changes - D",
							"Timbral similarity", "Timbral evolution - C", "Timbral evolution - D",
							"Global spectral overlap", "Frequency range overlap",
							"Rhythmic similarity", "Rhythmic density",
							"Harmonic function similarity", "Harmonic function transitions - C", "Harmonic function transitions - D"]

		# NON MODIFIABLE DYNAMICALLY
		self.N_CHROMA = 12
		self.N_MELBANDS = 40
		self.N_SPECTRALSHAPE = 7
		self.N_LOUDNESS = 2
		self.N_PITCH = 2
		self.N_ONSET = 1
		self.N_TONNETZ = 6 # COMPUTED IN PYTHON

		# INITIALIZE STATE VARIABLES
		# sequence feature vectors
		self.chroma_sequence = np.zeros((self.N_CHROMA, self.N_FFT_FRAMES))
		self.tonnetz_sequence = np.zeros((self.N_TONNETZ, self.N_FFT_FRAMES))
		self.melbands_sequence = np.zeros((self.N_MELBANDS, self.N_FFT_FRAMES))
		self.spectralshape_sequence = np.zeros((self.N_SPECTRALSHAPE, self.N_FFT_FRAMES))
		self.loudness_sequence = np.zeros((self.N_LOUDNESS, self.N_FFT_FRAMES))
		self.pitch_sequence = np.zeros((self.N_PITCH, self.N_FFT_FRAMES))
		self.onsets_sequence = []
		self.binaryRhythms_sequence = []
		# context feature vectors
		self.chroma_context = np.zeros((self.N_CHROMA, self.N_FFT_FRAMES))
		self.tonnetz_context = np.zeros((self.N_TONNETZ, self.N_FFT_FRAMES))
		self.melbands_context = np.zeros((self.N_MELBANDS, self.N_FFT_FRAMES))
		self.spectralshape_context = np.zeros((self.N_SPECTRALSHAPE, self.N_FFT_FRAMES))
		self.loudness_context = np.zeros((self.N_LOUDNESS, self.N_FFT_FRAMES))
		self.pitch_context = np.zeros((self.N_PITCH, self.N_FFT_FRAMES))
		self.onsets_context = []
		self.binaryRhythms_context = []

		# OTHER STATE VARIABLES
		self.ACTIVE = False
		self.REPETITION_NUMBER = 0 # how many times this loop has been going
		self.LAST_SATISFACTION_DEGREE = 0

		# checking features received
		self.EXPECTED_NUM_FEATURES = self.N_CHROMA + self.N_MELBANDS + self.N_SPECTRALSHAPE + self.N_LOUDNESS + self.N_ONSET + self.N_PITCH
		self.featuresInCounter_sequence = 0
		self.featuresInCounter_context = 0

		# compute candidate segment divisions
		self.candidate_segments_divisions = []
		for n in range(self.MIN_BEATS_PER_LOOP, self.BEATS_PER_LOOP, self.MIN_BEATS_PER_LOOP):
			if self.BEATS_PER_LOOP % n == 0:
				self.candidate_segments_divisions.append(n)
		self.candidate_segments_divisions.append(self.BEATS_PER_LOOP)
		self.min_loop_division = self.candidate_segments_divisions[0]
		print(f'Candidate segment divisions: {self.candidate_segments_divisions}')		

	def updateState(self, decision):

		if decision == 'A' and self.REPETITION_NUMBER > self.MIN_REPETITIONS:
			outcome = 'A'
			print(f'Decision A_{self.loop_track_num} ---> Segment selected for loop {self.loop_track_num}')
			self.REPETITION_NUMBER = 0
		elif self.REPETITION_NUMBER >= self.MAX_REPETITIONS:
			# DROP LOOP
			print(f'Decision Z_{self.loop_track_num} ---> Clearing loop {self.loop_track_num} audio buffer')
			outcome = 'Z'
			# send loop drop
			self.REPETITION_NUMBER = 0
		else:
			print(f'Decision R_{self.loop_track_num} ---> No update on loop {self.loop_track_num}')
			outcome = 'R'
			self.REPETITION_NUMBER += 1

		return outcome

	def computeSatsifactionCoefficient(self):

		sumOfLoops_sequence_descriptors = self.getContextState()

		# compute candidate segments based on segment divisions
		candidate_segments_descriptors = []
		for n in self.candidate_segments_divisions:
			candidate_segment_descriptors = self.getSequenceStateWDivisions(n)
			candidate_segments_descriptors.append(candidate_segment_descriptors)

		# Check for multiple candidate segments
		candidates_satisfaction_degrees = []
		# compute rules for each candidate segment
		for segment_descriptors in candidate_segments_descriptors:
			# COMPUTE COMPARISON METRICS
			comparison_metrics = self.compareSequenceWithLoops(segment_descriptors, sumOfLoops_sequence_descriptors, self.RHYTHM_SUBDIVISIONS)
			# EVALUATE LOOPING RULES
			rules_satisfied, rules_satisfaction_degree = self.evaluateLoopingRules(self.looping_rules, comparison_metrics)
			cumulative_satisfaction_degree = sum(rules_satisfaction_degree)/len(rules_satisfaction_degree)
			# set satisfaction degree to 0 if rules not satisfied
			if all(rules_satisfied):
				candidates_satisfaction_degrees.append(cumulative_satisfaction_degree)
			else:
				candidates_satisfaction_degrees.append(0)

		max_candidates_satisfaction_degree = np.argmax(np.array(candidates_satisfaction_degrees))
		loop_satisfaction_degree = candidates_satisfaction_degrees[max_candidates_satisfaction_degree]
		loop_rules_satisfied = True if candidates_satisfaction_degrees[max_candidates_satisfaction_degree] != 0 else False
		selected_candidate_num = max_candidates_satisfaction_degree
		
		# print(f'Loop track L_{self.loop_track_num}')
		# print(f'Most satisfactory candidate is segment {max_candidates_satisfaction_degree+1}')
		# print(f'Rule satisfaction degree {loop_satisfaction_degree:.3f}')

		return loop_satisfaction_degree, loop_rules_satisfied, selected_candidate_num


	def getBinaryRhythm(self, onsets):

		interval_size = int(self.N_BAR_SAMPLES / self.RHYTHM_SUBDIVISIONS)
		binary_rhythm = []
		for i in range(0, self.N_BAR_SAMPLES, interval_size):
			# if there is a onset in the bar division 1, otherwise 0
			flag = 0
			flag_dynamic = 0
			for onset in onsets:
				if onset > i and onset <= i+interval_size:
					flag = 1
			binary_rhythm.append(flag)
		return binary_rhythm


	def getContextState(self):

		# process features
		# current sequence features
		#onsets_seq = self.onsets_context
		onsets_seq = np.array([librosa.samples_to_frames(samp, hop_length=self.FFT_HOP_SIZE, n_fft=self.FFT_WINDOW) for samp in self.onsets_context])
		binary_rhythm_seq = self.binaryRhythms_context
		CQT_seq = self.melbands_context
		CQT_mean = CQT_seq.mean(axis=1)
		CQT_center_of_mass_seq = sum([CQT_mean[i]*i for i in range(CQT_mean.shape[0])]) / sum(CQT_mean)
		CQT_var_seq = np.std([CQT_mean[i]*i for i in range(CQT_mean.shape[0])])
		chroma_seq = self.chroma_context[:,:]
		discretechroma_seq = np.array([chroma_seq[:,int(j)] for j in onsets_seq])
		tonnetz_seq = self.tonnetz_context[:,:]
		discretetonnetz_seq = np.array([tonnetz_seq[:,int(j)] for j in onsets_seq])
		loudness_seq = self.loudness_context[0,:]
		discreteloudness_seq = np.array([loudness_seq[int(j)] for j in onsets_seq])
		centroid_seq = self.pitch_context[0,:]
		discretecentroid_seq = np.array([centroid_seq[int(j)] for j in onsets_seq])
		flatness_seq = self.spectralshape_context[5,:]
		discreteflatness_seq = np.array([flatness_seq[int(j)] for j in onsets_seq])

		bar_sequence_descriptors = [binary_rhythm_seq, np.array(onsets_seq), 
									CQT_seq, CQT_center_of_mass_seq, CQT_var_seq,
									chroma_seq, discretechroma_seq,
									loudness_seq.reshape(1,-1), discreteloudness_seq,
									centroid_seq.reshape(1,-1), discretecentroid_seq,
									flatness_seq.reshape(1,-1), discreteflatness_seq,
									tonnetz_seq, discretetonnetz_seq]
		
		return bar_sequence_descriptors

	def getSequenceStateWDivisions(self, n=1):

		segment_length_frames = int(self.frames_per_beat*n)+1

		# process features
		# current sequence features
		#onsets_seq = self.onsets_sequence
		onsets_seq = np.array([librosa.samples_to_frames(samp, hop_length=self.FFT_HOP_SIZE, n_fft=self.FFT_WINDOW) for samp in self.onsets_sequence])
		binary_rhythm_seq = self.binaryRhythms_sequence

		# CQT
		candidate_segment = np.zeros((self.N_MELBANDS, self.N_FFT_FRAMES))
		segment = self.melbands_sequence[:,-segment_length_frames:]
		#num_repetitions = int(self.BEATS_PER_LOOP / n)
		for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))+1):
			candidate_segment[:,k*segment_length_frames:(k+1)*segment_length_frames] = np.array(segment)
		CQT_seq = candidate_segment.copy()
		CQT_mean = CQT_seq.mean(axis=1)
		CQT_center_of_mass_seq = sum([CQT_mean[i]*i for i in range(CQT_mean.shape[0])]) / sum(CQT_mean)
		CQT_var_seq = np.std([CQT_mean[i]*i for i in range(CQT_mean.shape[0])])
		
		# chroma
		candidate_segment = np.zeros((self.N_CHROMA, self.N_FFT_FRAMES))
		segment = self.chroma_sequence[:,-segment_length_frames:]
		#num_repetitions = int(self.BEATS_PER_LOOP / n)
		for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))+1):
			candidate_segment[:,k*segment_length_frames:(k+1)*segment_length_frames] = np.array(segment)
		chroma_seq = candidate_segment.copy()
		discretechroma_seq = np.array([chroma_seq[:,j] for j in onsets_seq])

		# tonnetz
		candidate_segment = np.zeros((self.N_TONNETZ, self.N_FFT_FRAMES))
		segment = self.tonnetz_sequence[:,-segment_length_frames:]
		#num_repetitions = int(self.BEATS_PER_LOOP / n)
		for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))+1):
			candidate_segment[:,k*segment_length_frames:(k+1)*segment_length_frames] = np.array(segment)
		tonnetz_seq = candidate_segment.copy()
		discretetonnetz_seq = np.array([tonnetz_seq[:,j] for j in onsets_seq])
		
		# loudness
		candidate_segment = np.zeros((self.N_LOUDNESS, self.N_FFT_FRAMES))
		segment = self.loudness_sequence[:,-segment_length_frames:]
		#num_repetitions = int(self.BEATS_PER_LOOP / n)
		for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))+1):
			candidate_segment[:,k*segment_length_frames:(k+1)*segment_length_frames] = np.array(segment)
		loudness_seq = candidate_segment.copy()[0,:]
		discreteloudness_seq = np.array([loudness_seq[j] for j in onsets_seq])

		# centroid
		candidate_segment = np.zeros((self.N_SPECTRALSHAPE, self.N_FFT_FRAMES))
		segment = self.spectralshape_sequence[:,-segment_length_frames:]
		#num_repetitions = int(self.BEATS_PER_LOOP / n)
		for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))+1):
			candidate_segment[:,k*segment_length_frames:(k+1)*segment_length_frames] = np.array(segment)
		
		centroid_seq = candidate_segment.copy()[0,:]
		discretecentroid_seq = np.array([centroid_seq[j] for j in onsets_seq])

		flatness_seq = candidate_segment.copy()[5,:]
		discreteflatness_seq = np.array([flatness_seq[j] for j in onsets_seq])


		bar_sequence_descriptors = [binary_rhythm_seq, np.array(onsets_seq), 
									CQT_seq, CQT_center_of_mass_seq, CQT_var_seq,
									chroma_seq, discretechroma_seq,
									loudness_seq.reshape(1,-1), discreteloudness_seq,
									centroid_seq.reshape(1,-1), discretecentroid_seq,
									flatness_seq.reshape(1,-1), discreteflatness_seq,
									tonnetz_seq, discretetonnetz_seq]
		
		return bar_sequence_descriptors

	def evaluateLoopingRules(self, looping_rules, comparison_metrics):

		# EVALUATE LOOP CANDIDATE BASED ON RULES COMBINATION
		n_rule_components = len(looping_rules)
		rules_satisfied = []
		rules_satisfaction_degree = [] # between 0 and 1
		# iterate over rule components
		for rule in looping_rules:
			rule_satisfied = False
			rule_satisfaction_degree = 0
			threshold = rule["rule-threshold"]

			for i in range(len(self.RULE_NAMES)):
				if rule["rule-name"] == self.RULE_NAMES[i]:
					if rule["rule-type"] == "more":
						if comparison_metrics[i] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(comparison_metrics[i] - threshold)
					elif rule["rule-type"] == "less":
						if comparison_metrics[i] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - comparison_metrics[i])

			rules_satisfied.append(rule_satisfied)
			rules_satisfaction_degree.append(rule_satisfaction_degree)

		return rules_satisfied, rules_satisfaction_degree
	
	def compareSequenceWithLoops(self, bar_sequence_descriptors, sumOfLoops_sequence_descriptors, rhythm_subdivisions):

		if self.verbose == 2:
			print('Binary rhythms:')
		binary_comparison_coefficient, rhythm_density_coefficient = self.compareBinaryRhythms(bar_sequence_descriptors[0], sumOfLoops_sequence_descriptors[0], rhythm_subdivisions)

		## SPECTRAL BANDWIDTH
		if self.verbose == 2:
			print('Spectral bandwidth:')
		spectral_energy_overlap_coefficient, spectral_energy_difference_coefficient = self.compareSpectralBandwidth(bar_sequence_descriptors[2], bar_sequence_descriptors[3], bar_sequence_descriptors[4], 
																													sumOfLoops_sequence_descriptors[2], sumOfLoops_sequence_descriptors[3], sumOfLoops_sequence_descriptors[4])

		## CHROMA
		if self.verbose == 2:
			print('Chroma:')
		chroma_AE = self.computeTwodimensionalAE(bar_sequence_descriptors[5], sumOfLoops_sequence_descriptors[5])
		_, chroma_continuous_correlation = self.computeTwodimensionalContinuousCorrelation(bar_sequence_descriptors[5], sumOfLoops_sequence_descriptors[5])
		_, chroma_discrete_correlation = self.computeTwodimensionalDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[6], 
																					sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[6])

		## TONNETZ
		if self.verbose == 2:
			print('Tonnetz:')
		tonnetz_AE = self.computeTwodimensionalAE(bar_sequence_descriptors[13], sumOfLoops_sequence_descriptors[13])
		_, tonnetz_continuous_correlation = self.computeTwodimensionalContinuousCorrelation(bar_sequence_descriptors[13], sumOfLoops_sequence_descriptors[13])
		_, tonnetz_discrete_correlation = self.computeTwodimensionalDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[14], 
																					sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[14])

		## LOUDNESS
		if self.verbose == 2:
			print('Loudness:')
		loudness_MSE = self.computeSignalsMSE(bar_sequence_descriptors[7], sumOfLoops_sequence_descriptors[7])
		_, loudness_continuous_correlation = self.computeContinuousCorrelation(bar_sequence_descriptors[7], sumOfLoops_sequence_descriptors[7])
		_, loudness_discrete_correlation = self.computeDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[8], 
																	sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[8])

		## CENTROID
		if self.verbose == 2:
			print('Spectral centroid:')
		centroid_MSE = self.computeSignalsMSE(bar_sequence_descriptors[9], sumOfLoops_sequence_descriptors[9])
		_, centroid_continuous_correlation = self.computeContinuousCorrelation(bar_sequence_descriptors[9], sumOfLoops_sequence_descriptors[9])
		_, centroid_discrete_correlation = self.computeDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[10], 
																	sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[10])

		## FLATNESS
		if self.verbose == 2:
			print('Spectral flatness:')
		flatness_MSE = self.computeSignalsMSE(bar_sequence_descriptors[11], sumOfLoops_sequence_descriptors[11])
		_, flatness_continuous_correlation = self.computeContinuousCorrelation(bar_sequence_descriptors[11], sumOfLoops_sequence_descriptors[11])
		_, flatness_discrete_correlation = self.computeDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[12], 
																	sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[12])
		if self.verbose == 2:
			print()

		# these have to match the order in self.RULE_NAMES
		comparison_metrics = [chroma_AE, chroma_continuous_correlation/2+0.5, chroma_discrete_correlation/2+0.5,
							centroid_MSE, centroid_continuous_correlation/2+0.5, centroid_discrete_correlation/2+0.5,
							loudness_MSE, loudness_continuous_correlation/2+0.5, loudness_discrete_correlation/2+0.5,
							flatness_MSE, flatness_continuous_correlation/2+0.5, flatness_discrete_correlation/2+0.5,
							spectral_energy_difference_coefficient, spectral_energy_overlap_coefficient,
							binary_comparison_coefficient, rhythm_density_coefficient]

		return comparison_metrics


	# FUNCTIONS TO COMPARE FEATURES
	def compareBinaryRhythms(self, binary_rhythm1, binary_rhythm2, rhythm_subdivisions=16):
		if self.verbose == 2:
			print(np.array(binary_rhythm1))
			print(np.array(binary_rhythm2))
		binary_comparison = [1 if binary_rhythm1[i] == binary_rhythm2[i] else 0 for i in range(len(binary_rhythm1))]
		binary_comparison_coefficient = sum(binary_comparison) / rhythm_subdivisions 
		if self.verbose == 2:
			print(f'Binary Comparison coefficient: {binary_comparison_coefficient:.3f}')
		rhythm_density_coefficient = abs(np.array(binary_rhythm1).sum() - np.array(binary_rhythm2).sum()) / rhythm_subdivisions
		if self.verbose == 2:
			print(f"Rhythm Density Comparison coefficient: {rhythm_density_coefficient:.3f}")
		return binary_comparison_coefficient, rhythm_density_coefficient

	def compareSpectralBandwidth(self, CQT1, CQT1_center_of_mass, CQT1_var, CQT2, CQT2_center_of_mass, CQT2_var, plotflag=False):
		CQT1_mean = CQT1.mean(axis=1)
		CQT2_mean = CQT2.mean(axis=1)
		spectral_energy_overlap_index = max(0, 
		    min(CQT1_center_of_mass+CQT1_var, CQT2_center_of_mass+CQT2_var) - max(CQT1_center_of_mass-CQT1_var, CQT2_center_of_mass-CQT2_var))
		spectral_energy_overlap_coefficient = min(spectral_energy_overlap_index, min(CQT1_var*2, CQT2_var*2)) / min(CQT1_var*2, CQT2_var*2)
		if self.verbose == 2:
			print(f"Spectral energy overlap coefficient: {spectral_energy_overlap_coefficient:.3f}")
		spectral_energy_difference_coefficient = np.abs(CQT1_mean - CQT2_mean).mean()
		if self.verbose == 2:
			print(f"Spectral energy difference coefficient: {spectral_energy_difference_coefficient:.3f}")
		return spectral_energy_overlap_coefficient, spectral_energy_difference_coefficient

	def computeSignalsMSE(self, signal1, signal2):
		minsignal = min(signal1.min(), signal2.min())
		maxsignal = max(signal1.max(), signal2.max())
		normalized_signal1 = (signal1 - minsignal) / (maxsignal - minsignal)
		normalized_signal2 = (signal2 - minsignal) / (maxsignal - minsignal)
		MSE = ((normalized_signal1 - normalized_signal2)**2).mean()
		if self.verbose == 2:
			print(f'MSE between the two signal is: {MSE:.3f}')
		return MSE

	def computeContinuousCorrelation(self, signal1, signal2):
		if (signal1.max() - signal1.min()) != 0 and (signal2.max() - signal2.min()) != 0:
			normalized_signal1 = (signal1 - signal1.min()) / (signal1.max() - signal1.min())
			normalized_signal2 = (signal2 - signal2.min()) / (signal2.max() - signal2.min())
			signal_time_correlation = normalized_signal1 * normalized_signal2
			time_correlation_coefficient = np.mean(signal_time_correlation.sum(axis=1) / signal_time_correlation.shape[1])
			pearson_correlation = ((signal1 - signal1.mean()) * (signal2 - signal2.mean()) / (signal1.std() * signal2.std())).mean()
			if np.isnan(time_correlation_coefficient):
				time_correlation_coefficient = 0
			if np.isnan(pearson_correlation):
				pearson_correlation = 0
			if self.verbose == 2:
				print(f"Continuous pearson correlation coefficient: {pearson_correlation:.3f}")
		else:
			time_correlation_coefficient = 0
			pearson_correlation = 0
			if self.verbose == 2:
				print(f"Continuous pearson correlation coefficient: {pearson_correlation:.3f}")
		return time_correlation_coefficient, pearson_correlation

	def computeDiscreteCorrelation(self, onsets1, values1, onsets2, values2, plotflag=False):
		if onsets1.shape[0] != 0 and onsets2.shape[0] != 0:
			values1 = np.append(values1, values1[-1])
			values2 = np.append(values2, values2[-1])
			# make arrays same size
			both_onsets = np.unique(np.sort(np.concatenate((onsets1, onsets2), axis=0)))
			# join arrays
			dtype = [('key', int), ('field', float)]
			x = np.array([(onsets1[i], values1[i]) for i in range(onsets1.shape[0])], dtype=dtype)
			y = np.array([(onsets2[i], values2[i]) for i in range(onsets2.shape[0])], dtype=dtype)
			join = np.lib.recfunctions.join_by('key', x, y, jointype='outer')
			join.fill_value = 1e10
			join = join.filled()
			X = [value[1] for value in join]
			for i in range(len(X)):
			    if X[i] != 1e10:
			        X[i] = X[i]
			    else:
			        X[i] = X[i-1]
			Y = [value[2] for value in join]
			for i in range(len(Y)):
			    if Y[i] != 1e10:
			        Y[i] = Y[i]
			    else:
			        Y[i] = Y[i-1]
			# join arrays
			new_values1 = np.array(X)
			new_values2 = np.array(Y)
			normalized_values1 = (new_values1 - new_values1.min()) / (new_values1.max() - new_values1.min())
			normalized_values2 = (new_values2 - new_values2.min()) / (new_values2.max() - new_values2.min())
			normalized_values_time_correlation = normalized_values1 * normalized_values2				
			discrete_time_correlation_coefficient = np.mean(normalized_values_time_correlation.sum() / normalized_values_time_correlation.shape[0])
			discrete_pearson_correlation = ((new_values1 - new_values1.mean()) * (new_values2 - new_values2.mean()) / (new_values1.std() * new_values2.std())).mean()
			if np.isnan(discrete_time_correlation_coefficient):
				discrete_time_correlation_coefficient = 0
			if np.isnan(discrete_pearson_correlation):
				discrete_pearson_correlation = 0
		else: 
			discrete_time_correlation_coefficient = 0
			discrete_pearson_correlation = 0
		if self.verbose == 2:
			print(f"Discrete pearson correlation coefficient: {discrete_pearson_correlation:.3f}")

		return discrete_time_correlation_coefficient, discrete_pearson_correlation

	def computeTwodimensionalMSE(self, signal1, signal2):
		minsignal = np.concatenate((signal1.min(axis=1).reshape(-1,1), signal2.min(axis=1).reshape(-1,1)), axis=1).min(axis=1).reshape(-1,1)
		maxsignal = np.concatenate((signal1.max(axis=1).reshape(-1,1), signal2.max(axis=1).reshape(-1,1)), axis=1).max(axis=1).reshape(-1,1)
		normalized_signal1 = (signal1 - minsignal) / (maxsignal - minsignal)
		normalized_signal2 = (signal2 - minsignal) / (maxsignal - minsignal)
		MSE = ((normalized_signal1 - normalized_signal2)**2).mean(axis=1).mean()
		if self.verbose == 2:
			print(f'MSE between the two signal is: {MSE:.3f}')
		return MSE

	def computeTwodimensionalAE(self, signal1, signal2, plotflag=False):
		similarity_coefficient = np.abs(np.mean(signal1.mean(axis=1) - signal2.mean(axis=1)))
		if self.verbose == 2:
			print(f"Absolute Error difference coefficient: {similarity_coefficient:.3f}")
		return similarity_coefficient

	def computeTwodimensionalContinuousCorrelation(self, signal1, signal2, plotflag=False):
		time_correlation = signal1 * signal2
		time_correlation_coefficient = np.mean(time_correlation.sum(axis=1) / time_correlation.shape[1])
		continuous_pearson_correlation = ((signal1 - signal1.mean(axis=1).reshape(-1,1)) * (signal2 - signal2.mean(axis=1).reshape(-1,1))).mean(axis=1) / (signal1.std(axis=1) * signal2.std(axis=1))
		if np.isnan(np.sum(time_correlation_coefficient)):
			time_correlation_coefficient = np.array([0])
		if np.isnan(np.sum(continuous_pearson_correlation)):
			continuous_pearson_correlation = np.array([0])
		if self.verbose == 2:
			print(f"Continuous pearson correlation coefficient: {continuous_pearson_correlation.mean():.3f}")
		return time_correlation_coefficient, continuous_pearson_correlation.mean()

	def computeTwodimensionalDiscreteCorrelation(self, onsets1, values1, onsets2, values2):
		if onsets1.shape[0] == 0 or onsets2.shape[0] == 0:
			discrete_time_correlation_coefficient = np.array([0])
			discrete_pearson_correlation = np.array([0])
		else:
			try:
				# this is due to a bug in np.lib.recfunctions.join_by() with a list in 'field' 
				values1 = np.concatenate((values1, values1[-1,:].reshape(1,-1)), axis=0)
				values2 = np.concatenate((values2, values2[-1,:].reshape(1,-1)), axis=0)
				# make arrays same size
				both_onsets = np.unique(np.sort(np.concatenate((onsets1, onsets2), axis=0)))
				dtype = [('key', int), ('field', list)]
				x = np.array([(int(onsets1[i]), values1[i,:].tolist()) for i in range(onsets1.shape[0])], dtype=dtype)
				y = np.array([(int(onsets2[i]), values2[i,:].tolist()) for i in range(onsets2.shape[0])], dtype=dtype)
				join = np.lib.recfunctions.join_by('key', x, y, jointype='outer')
				join.fill_value = 1e10
				join = join.filled()
				X = [value[1] for value in join]
				for i in range(len(X)):
				    if X[i] != 1e10:
				        X[i] = X[i]
				    else:
				        X[i] = X[i-1]
				Y = [value[2] for value in join]
				for i in range(len(Y)):
				    if Y[i] != 1e10:
				        Y[i] = Y[i]
				    else:
				        Y[i] = Y[i-1]
				new_values1 = np.array(X)
				new_values2 = np.array(Y)
				values_time_correlation = new_values1 * new_values2
				discrete_time_correlation_coefficient = np.mean(values_time_correlation.sum(axis=0) / values_time_correlation.shape[0])
				discrete_pearson_correlation = ((new_values1 - new_values1.mean(axis=1).reshape(-1,1)) * (new_values2 - new_values2.mean(axis=1).reshape(-1,1))).mean(axis=1) / (new_values1.std(axis=1) * new_values2.std(axis=1))
				if np.isnan(np.sum(discrete_time_correlation_coefficient)):
					discrete_time_correlation_coefficient = np.array([0])
				if np.isnan(np.sum(discrete_pearson_correlation)):
					discrete_pearson_correlation = np.array([0])
			except:
				if self.verbose == 2:
					print('COULD NOT COMPUTE CORRELATION')
				discrete_time_correlation_coefficient = np.array([0])
				discrete_pearson_correlation = np.array([0])
			if self.verbose == 2:
				print(f"Discrete pearson correlation coefficient: {discrete_pearson_correlation.mean():.3f}")
		return discrete_time_correlation_coefficient, discrete_pearson_correlation.mean()



class MultiAgentAutonomousLooperOnline():

	def __init__(self, 
			sr=44100,
			fft_window=1024,
			fft_hopSize=512,
			ip="127.0.0.1", # localhost
			port_snd=6667, # send port to PD
			port_rcv=6666 # receive port from PD
		):

		print()
		print('Initializing Multi-Agent Autonomous Looper online')
		print('-'*50)
		print()


		# looper state variables
		self.decision_elements = []
		self.decision_log = []
		self.LOOP_AGENTS = {}

		# updated from PD
		self.BEATS_PER_LOOP = 8
		self.MIN_BEATS_PER_LOOP = 4
		self.BASE_BPM = 120
		self.RHYTHM_SUBDIVISIONS = 16
		self.MIN_REPETITIONS = 8
		self.MAX_REPETITIONS = 24

		# BUFFER SIZES CONFIGURATIONS
		self.sr = sr
		self.FFT_WINDOW = fft_window
		self.FFT_HOP_SIZE = fft_hopSize

		# SERVER
		self.ip = ip
		self.port_snd = port_snd # send port to PD
		self.port_rcv = port_rcv # receive port from PD

		## OSC SERVER
		# define dispatcher
		dispatcher = Dispatcher()
		dispatcher.map("/MOTHER_LOOPER", self.loopStart_handler)
		dispatcher.map("/HELLO", self.hello_handler)
		dispatcher.map("/RESET", self.reset_handler)
		dispatcher.map("/BEATS_PER_LOOP", self.beatsPerLoop_handler)
		dispatcher.map("/BASE_BPM", self.baseBpm_handler)
		dispatcher.map("/END_OF_SUBDIV", self.endOfSubdivs_handler)
		dispatcher.map("/features/*", self.liveFeaturesIn_handler)
		dispatcher.set_default_handler(self.default_handler)

		# define client
		self.client = udp_client.SimpleUDPClient(self.ip, self.port_snd)

		# define server
		server = BlockingOSCUDPServer((self.ip, self.port_rcv), dispatcher)
		server.serve_forever()  # Blocks forever

	def endOfSubdivs_handler(self, address, *args):
		print()
		print(f'End of subdivision: {int(args[0])}')
		print('-'*20)


	def loopStart_handler(self, address, *args):
		if args[0] == 1:
			print()
			print('Creating new decision log...')
			print()
			self.decision_log = []
		elif args[0] == 0:
			print()
			print('Saving decision log...')
			print()
			with open(f'performance/decisions_log.json', 'w', encoding='utf-8') as f:
				json.dump(self.decision_log, f, ensure_ascii=False, indent=4)
			performance_info = { "BASE_BPM": self.BASE_BPM, "BEATS_PER_LOOP": self.BEATS_PER_LOOP }
			with open(f'performance/performance_info.json', 'w', encoding='utf-8') as f:
				json.dump(performance_info, f, ensure_ascii=False, indent=4)

	def hello_handler(self, address, *args):
		# register loop agent
		newLoopAgent = LoopAgent(loop_track_num=int(args[0]),
								sr=self.sr,
								fft_window=self.FFT_WINDOW,
								fft_hopSize=self.FFT_HOP_SIZE,
								BEATS_PER_LOOP=self.BEATS_PER_LOOP,
								MIN_BEATS_PER_LOOP=self.MIN_BEATS_PER_LOOP,
								BASE_BPM=self.BASE_BPM, 
								RHYTHM_SUBDIVISIONS=self.RHYTHM_SUBDIVISIONS,
								MIN_REPETITIONS=self.MIN_REPETITIONS,
								MAX_REPETITIONS=self.MAX_REPETITIONS
							)
		self.LOOP_AGENTS[int(args[0])] = newLoopAgent
		self.loops_satisfaction_degrees[int(args[0])] = None
		self.loops_rules_satisfied[int(args[0])] = False
		self.loops_selected_subdivision_nums[int(args[0])] = None
		self.loops_updated[int(args[0])] = False
		print(f'Loop agents: {self.LOOP_AGENTS}')

	def reset_handler(self, address, *args):
		self.LOOP_AGENTS = {}
		self.loops_satisfaction_degrees = {}
		self.loops_rules_satisfied = {}
		self.loops_selected_subdivision_nums = {}
		self.loops_updated = {}
		print('RESET')

	def beatsPerLoop_handler(self, address, *args):
		self.BEATS_PER_LOOP = args[0]
		print(f"BEATS_PER_LOOP: {self.BEATS_PER_LOOP}")

	def baseBpm_handler(self, address, *args):
		self.BASE_BPM = args[0]
		print(f"BASE_BPM: {self.BASE_BPM}")

	def default_handler(self, address, *args):
		print(f"DEFAULT {address}: {args}")

	def liveFeaturesIn_handler(self, address, *args):
		#print(f"{address}: {len(args)}")
		feature_name = address.split('/')[-1].split('-')[0]
		loop_num = int(address.split('/')[2])
		signal_context = address.split('/')[3]
		feature_component_num = int(address.split('/')[-1].split('-')[-1])
		# print(f'rcvd: {loop_num}/{signal_context}/{feature_name}-{feature_component_num}')

		if self.LOOP_AGENTS:
			loopAgent = self.LOOP_AGENTS[loop_num]

			# SIMPLE FEATURE RECEIVER
			if signal_context == "signal":
				if feature_name == 'chroma':
					loopAgent.chroma_sequence[feature_component_num, :] = np.array(args)[:loopAgent.N_FFT_FRAMES]
					loopAgent.featuresInCounter_sequence += 1
				elif feature_name == 'spectralshape':
					loopAgent.spectralshape_sequence[feature_component_num, :] = np.array(args)[:loopAgent.N_FFT_FRAMES]
					loopAgent.featuresInCounter_sequence += 1
				elif feature_name == 'melbands':
					loopAgent.melbands_sequence[feature_component_num, :] = np.array(args)[:loopAgent.N_FFT_FRAMES]
					loopAgent.featuresInCounter_sequence += 1
				elif feature_name == 'loudness':
					loopAgent.loudness_sequence[feature_component_num, :] = np.array(args)[:loopAgent.N_FFT_FRAMES]
					loopAgent.featuresInCounter_sequence += 1
				elif feature_name == 'pitch':
					loopAgent.pitch_sequence[feature_component_num, :] = np.array(args)[:loopAgent.N_FFT_FRAMES]
					loopAgent.featuresInCounter_sequence += 1
				elif feature_name == 'onsets':
					loopAgent.onsets = np.abs(np.array(args))
					loopAgent.binaryRhythms_sequence = loopAgent.getBinaryRhythm(loopAgent.onsets)
					loopAgent.featuresInCounter_sequence += 1

			elif signal_context == "context":
				if feature_name == 'chroma':
					loopAgent.chroma_context[feature_component_num, :] = np.array(args)[:loopAgent.N_FFT_FRAMES]
					loopAgent.featuresInCounter_context += 1
				elif feature_name == 'spectralshape':
					loopAgent.spectralshape_context[feature_component_num, :] = np.array(args)[:loopAgent.N_FFT_FRAMES]
					loopAgent.featuresInCounter_context += 1
				elif feature_name == 'melbands':
					loopAgent.melbands_context[feature_component_num, :] = np.array(args)[:loopAgent.N_FFT_FRAMES]
					loopAgent.featuresInCounter_context += 1
				elif feature_name == 'loudness':
					loopAgent.loudness_context[feature_component_num, :] = np.array(args)[:loopAgent.N_FFT_FRAMES]
					loopAgent.featuresInCounter_context += 1
				elif feature_name == 'pitch':
					loopAgent.pitch_context[feature_component_num, :] = np.array(args)[:loopAgent.N_FFT_FRAMES]
					loopAgent.featuresInCounter_context += 1
				elif feature_name == 'onsets':
					loopAgent.onsets_context = np.abs(np.array(args))
					loopAgent.binaryRhythms_context = loopAgent.getBinaryRhythm(loopAgent.onsets_context)
					loopAgent.featuresInCounter_context += 1

			# ACTION WHEN ALL FEATURES HAVE BEEN RECIEVED
			if loopAgent.featuresInCounter_sequence >= loopAgent.EXPECTED_NUM_FEATURES and loopAgent.featuresInCounter_context >= loopAgent.EXPECTED_NUM_FEATURES:
				# print(f'Loop {loop_num}: all features recieved')
				loopAgent.featuresInCounter_sequence = 0
				loopAgent.featuresInCounter_context = 0

				# compute tonnetz from chroma
				loopAgent.tonnetz_sequence = librosa.feature.tonnetz(chroma=loopAgent.chroma_sequence, sr=loopAgent.sr)
				loopAgent.tonnetz_context = librosa.feature.tonnetz(chroma=loopAgent.chroma_context, sr=loopAgent.sr)

				loop_satisfaction_degree, loop_rules_satisfied, selected_candidate_num = loopAgent.computeSatsifactionCoefficient()
				self.loops_satisfaction_degrees[loop_num] = loop_satisfaction_degree
				self.loops_rules_satisfied[loop_num] = loop_rules_satisfied
				self.loops_selected_subdivision_nums[loop_num] = selected_candidate_num
				self.loops_updated[loop_num] = True


				if all(self.loops_updated.values()):
					# print('ALL COMPUTED')

					# make decision
					self.loops_satisfaction_degrees = collections.OrderedDict(sorted(self.loops_satisfaction_degrees.items()))
					self.loops_rules_satisfied = collections.OrderedDict(sorted(self.loops_rules_satisfied.items()))
					self.loops_selected_subdivision_nums = collections.OrderedDict(sorted(self.loops_selected_subdivision_nums.items()))

					# print(self.loops_satisfaction_degrees)
					# print(self.loops_rules_satisfied)
					# print(self.loops_selected_subdivision_nums)

					all_loops_satisfaction_degrees = [self.loops_satisfaction_degrees[i] if self.loops_rules_satisfied[i] else 0 for i in list(self.loops_satisfaction_degrees.keys())]
					loops_sorted_by_satisfaction_degree = np.argsort(np.array(all_loops_satisfaction_degrees)).tolist()
					MAX_A = 1 # MAXIMUM NUMBER OF DECISIONS PER SEGMENT
					A_counter = 0
					for i in range(len(loops_sorted_by_satisfaction_degree)):				
						if self.loops_rules_satisfied[i] and A_counter < MAX_A:
							decision_outcome = self.LOOP_AGENTS[i].updateState('A')
							if decision_outcome == 'A':
								self.client.send_message("/loopdecision/A", str(i))
								A_counter += 1
							elif decision_outcome == 'Z':
								self.client.send_message("/loopdecision/Z", str(i))
						else:
							decision_outcome = self.LOOP_AGENTS[i].updateState('R')

					# update state variables
					for key in self.loops_updated.keys():
						self.loops_updated[key] = False


if __name__ == '__main__': 

	looper = MultiAgentAutonomousLooperOnline()


