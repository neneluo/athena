from athena import get_wave_file_length
import sys, os
import pandas
import logging

_, train_list_file, dev_list_file, test_list_file = sys.argv
speaker_id_dict = {}

for list_file in (train_list_file, dev_list_file, test_list_file):
    files = []
    csv_file = list_file + ".csv"
    with open(list_file, "r") as LIST:
        for line in LIST.readlines():
            speaker_name, wav_file = line.strip().split()
            if speaker_name not in speaker_id_dict:
                num = len(speaker_id_dict)
                speaker_id_dict[speaker_name] = num
            wav_length = get_wave_file_length(wav_file)
            utt_key = speaker_name + "_" + wav_file.split("/")[-1].split(".")[0]
            files.append(
                (os.path.abspath(wav_file), wav_length, speaker_id_dict[speaker_name], speaker_name, utt_key)
            )

    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "speaker_id", "speaker_name", "utt_key"])
    df.to_csv(csv_file, index=False, sep="\t")
    print("Successfully generated csv file {}".format(csv_file))
