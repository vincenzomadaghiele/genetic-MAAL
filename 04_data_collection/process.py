import os
import json
import librosa
import matplotlib.pyplot as plt
import matplotlib.style as ms
import numpy as np
import colorsys


if __name__ == '__main__': 

	# load sound files from performance
	perf_filepath = './performance'
	dryin_filepath = f'{perf_filepath}/dry-in.wav'
	loops_paths = [path if path.split('.')[0].split('-')[0] == 'loop' else None for path in os.listdir(perf_filepath)]
	loops_paths = [x for x in loops_paths if x is not None]	
	loops_paths.sort() 

	print(loops_paths)

	# load decision log
	with open(f'{perf_filepath}/decisions_log.json', 'r') as file:
	    decisions_log = json.load(file)

	# load setup info
	with open(f'{perf_filepath}/info.json', 'r') as file:
	    info = json.load(file)
	BEATS_PER_LOOP = info["BEATS_PER_LOOP"]
	BASE_BPM = info["BASE_BPM"]

	# LOAD AUDIO TRACKS  
	sr = 44100
	MAX_SIGNAL_SIZE = 2000000000
	signal, sr = librosa.load(dryin_filepath, sr=sr, mono=True)
	signal = signal[:MAX_SIGNAL_SIZE]
	loops_audiotracks = []
	for loop_path in loops_paths: 
		loop_signal, _ = librosa.load(f'{perf_filepath}/{loop_path}', sr=sr, mono=True)
		loops_audiotracks.append(loop_signal[:MAX_SIGNAL_SIZE])

	# TEMPO
	tempo_bps = BASE_BPM / 60 # beats per second
	beat_seconds = 1 / tempo_bps # duration of one beat [seconds]
	BEAT_SAMPLES = int(beat_seconds * sr) # number of samples of one beat
	NUM_BEATS = int(signal.shape[0] / BEAT_SAMPLES) / 2 # number of beats in the track

	N_LOOPS = len(loops_paths)
	min_loop_division = 4
	signal_subdivided_samples = [(i * BEAT_SAMPLES * min_loop_division) for i in range(int(NUM_BEATS / min_loop_division))] # samples at which each looped bar starts

	loops_bars = [[] for _ in range(N_LOOPS)]
	for decision in decisions_log:
		if decision["decisions"][0]["decision_type"] == "A":
			loops_bars[int(decision["decisions"][0]["loop_track (i)"])].append(int(decision["subdivision_index (m)"]))


	# BUG!!!
	prov = loops_bars[-1]
	loops_bars[-1] = loops_bars[1]
	loops_bars[1] = prov


	# COMPUTE SUMMARY FIGURE
	fig, ax = plt.subplots(N_LOOPS+1, figsize=(12,N_LOOPS*2-2), gridspec_kw={'height_ratios': np.ones((N_LOOPS)).tolist().insert(0,3)})
	fig.subplots_adjust(top=0.8)

	colors = [colorsys.hsv_to_rgb(np.random.random(), 0.8, 0.9) for n in range(N_LOOPS)]
	vertical_line_length = np.max(signal)+ 0.1
	librosa.display.waveshow(signal, sr=sr, ax=ax[0], label='original signal', alpha=0.4)
	# ax[0].vlines(librosa.samples_to_time(signal_subdivided_samples), -1*vertical_line_length, vertical_line_length, color='black', alpha=0.9, linestyle='--', lw=0.8)
	# plot loops
	for n in range(N_LOOPS):
		librosa.display.waveshow(loops_audiotracks[n], sr=sr, ax=ax[n+1], label=f'loop {n+1}', alpha=0.7, color=colors[n])
	
		loop_in_sig = np.zeros_like(signal)
		for j, bar_num in enumerate(loops_bars[n]):
			num_samples_selected_segment = BEAT_SAMPLES * BEATS_PER_LOOP
			#num_samples_selected_segment = self.candidate_segments_divisions[loops_candidate_num[n][j]] * min_loop_division * BEAT_SAMPLES
			loop_in_sig[int(signal_subdivided_samples[bar_num-1]*2):int(signal_subdivided_samples[bar_num-1]*2)+num_samples_selected_segment] = signal[int(signal_subdivided_samples[bar_num-1]*2):int(signal_subdivided_samples[bar_num-1]*2)+num_samples_selected_segment]
			ax[0].vlines(librosa.samples_to_time(int(signal_subdivided_samples[bar_num-1]*2)+num_samples_selected_segment, sr=sr), -1*vertical_line_length, vertical_line_length, color=colors[n], alpha=0.9, linestyle='--', lw=0.8)
			ax[n+1].vlines(librosa.samples_to_time(int(signal_subdivided_samples[bar_num-1]*2)+num_samples_selected_segment, sr=sr), -1*vertical_line_length, vertical_line_length, color='black', alpha=0.8, linestyle='--', lw=0.8)
		ax[0].plot(librosa.samples_to_time(np.arange(0, signal.shape[0]), sr=sr), loop_in_sig, label=f'loop {n+1}', alpha=0.6, color=colors[n])
	
		# ax[n+1].vlines(librosa.samples_to_time(signal_subdivided_samples), -1*vertical_line_length, vertical_line_length, color='black', alpha=0.9, linestyle='--', lw=0.8)
		ax[n+1].xaxis.set_visible(False)
		ax[n+1].set_ylabel(f"$L_{n+1}$", rotation=0, ha='right', fontsize=13)
		ax[n+1].spines['right'].set_visible(False)
		ax[n+1].spines['top'].set_visible(False)
		ax[n+1].spines['bottom'].set_visible(False)
		if n != N_LOOPS:
			ax[n].set_xticklabels([])
		ax[n+1].set_yticklabels([])

	ax[n+1].xaxis.set_visible(True) # set last axis visible
	ax[n+1].set_xlabel("Time $[mm:ss]$")
	ax[n+1].spines['bottom'].set_visible(True)

	ax[0].spines['right'].set_visible(False)
	ax[0].spines['top'].set_visible(False)
	ax[0].spines['bottom'].set_visible(False)
	ax[0].set_xticklabels([])
	ax[0].set_yticklabels([])

	ax[0].set_ylabel("$x(t)$", rotation=0, ha='right', fontsize=13)
	ax[0].xaxis.set_visible(False)
	fig.suptitle('LOOP PERFORMANCE SUMMARY', size=16, y=0.9)
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig(f'{perf_filepath}/loops_figure.png')
	plt.show()


	# cut quantized version based on times
	# export resulting decision log and audio files
	def cutPerformance(start=None, stop=None):
		return





