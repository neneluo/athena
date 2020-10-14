import os, sys

if len(sys.argv) != 5:
	print("Usage: python replace_wav_to_fbank.py in.csv wav_scp fbank_scp out.csv")
	sys.exit()

_, in_file, wav_file, fbank_file, out_file = sys.argv

out = open(out_file, "w")

wav_dict, fbank_dict = {}, {}
lines = open(wav_file, "r").read().splitlines()
for line in lines:
	key, path = line.strip().split(" ")
	wav_dict[key] = path

lines = open(fbank_file, "r").read().splitlines()
for line in lines:
	key, path = line.strip().split(" ")
	fbank_dict[wav_dict[key]] = path

lines = open(in_file, "r").read().splitlines()
out.write(lines[0]+"\n")
for line in lines[1:]:
	path, a = line.split("\t", 1)
	out.write(fbank_dict[path] + "\t" + a + "\n")
out.close()
