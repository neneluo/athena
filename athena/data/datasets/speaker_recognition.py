# coding=utf-8
# Copyright (C) 2020 ATHENA AUTHORS; Ne Luo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=no-member, invalid-name
""" speaker dataset """

import os
import random
from absl import logging
import tensorflow as tf
import librosa
import numpy as np
from .base import SpeechBaseDatasetBuilder

class SpeakerRecognitionDatasetBuilder(SpeechBaseDatasetBuilder):
    """SpeakerRecognitionDatasetBuilder
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "cut_frame": [None],
        "input_length_range": [20, 50000],
        "data_csv": None
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        if self.hparams.data_csv is not None:
            self.preprocess_data(self.hparams.data_csv)

    def preprocess_data(self, file_path):
        """ Generate a list of tuples
            (wav_filename, wav_length_ms, speaker_id, speaker_name).
        """
        logging.info("Loading data csv {}".format(file_path))
        with open(file_path, "r", encoding='utf-8') as data_csv:
            lines = data_csv.read().splitlines()[1:]
        lines = [line.split("\t", 3) for line in lines]
        lines.sort(key=lambda item: int(item[1]))
        self.entries = [tuple(line) for line in lines]
        self.speakers = list(set(line[-1] for line in lines))

        # apply input length filter
        self.entries = list(filter(lambda x: int(x[1]) in
                            range(self.hparams.input_length_range[0],
                            self.hparams.input_length_range[1]), self.entries))
        return self

    def cut_features(self, feature):
        """cut acoustic featuers
        """
        min_len, max_len = self.hparams.cut_frame
        # length = self.hparams.cut_frame
        length = tf.random.uniform([], min_len, max_len, tf.int32)
        # randomly select a start frame
        max_start_frames = tf.shape(feature)[0] - length
        if max_start_frames <= 0:
            return feature
        start_frames = tf.random.uniform([], 0, max_start_frames, tf.int32)
        return feature[start_frames:start_frames + length, :, :]

    def load_data_librosa(self, path, win_length=400, sr=16000, hop_length=160,
                n_fft=512):
        wav, _ = librosa.load(path, sr=sr)
        wav = np.asfortranarray(wav)
        linear_spect = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        mag, _ = librosa.magphase(linear_spect)  # magnitude (257, time_steps)
        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(mag, 1, keepdims=True)
        std = np.std(mag, 1, keepdims=True)
        data = (mag - mu) / (std + 1e-5)
        return data

    def __getitem__(self, index):
        #audio_data, _, spkid, spkname = self.entries[index]
        audio_data = self.entries[index][0]
        spkid = self.entries[index][2]
        #spk_name = self.entries[index][3]
        #utt_key = self.entries[index][4]
        feat = self.audio_featurizer(audio_data)
        #feat = self.feature_normalizer(feat, 'global')
        #mu = np.mean(feat, 1, keepdims=True)
        #std = np.std(feat, 1, keepdims=True)
        #feat = (feat - mu) / (std + 1e-5)
        #feat = self.load_data_librosa(audio_data)
        feat = tf.reshape(tf.convert_to_tensor(feat), [-1, 40, 1])
        if self.hparams.cut_frame != [None]:
            feat = self.cut_features(feat)
        feat_length = feat.shape[0]
        spkid = [spkid]
        return {
            "input": feat,
            "input_length": feat_length,
            "output_length": 1,
            "output": spkid
        }

    def __len__(self):
        ''' return the number of data samples '''
        return len(self.entries)

    @property
    def num_class(self):
        ''' return the number of speakers'''
        return len(self.speakers)

    @property
    def speaker_list(self):
        return self.speakers

    @property
    def sample_type(self):
        """:obj:`@property`

        Returns:
            dict: sample_type of the dataset::

            {
                "input": tf.float32,
                "input_length": tf.int32,
                "output_length": tf.int32,
                "output": tf.int32
            }
        """
        return {
            "input": tf.float32,
            "input_length": tf.int32,
            "output_length": tf.int32,
            "output": tf.int32
        }

    @property
    def sample_shape(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "input": tf.TensorShape([None, dim, nc]),
                "input_length": tf.TensorShape([]),
                "output_length": tf.TensorShape([]),
                "output": tf.TensorShape([None])
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return {
            "input": tf.TensorShape([None, dim, nc]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None])
        }

    @property
    def sample_signature(self):
        """:obj:`@property`

        Returns:
            dict: sample_signature of the dataset::

            {
                "input": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return (
            {
                "input": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            },
        )


class SpeakerVerificationDatasetBuilder(SpeakerRecognitionDatasetBuilder):
    """SpeakerVerificationDatasetBuilder
    """

    def __init__(self, config=None):
        super().__init__(config=config)

    def preprocess_data(self, file_path):
        """ Generate a list of tuples
            (wav_filename_a, speaker_a, wav_filename_b, speaker_b, label).
        """
        logging.info("Loading data csv {}".format(file_path))
        with open(file_path, "r", encoding='utf-8') as data_csv:
            lines = data_csv.read().splitlines()[1:]
        lines = [line.split("\t", 4) for line in lines]
        self.entries = [tuple(line) for line in lines]
        return self

    def __getitem__(self, index):
        """get a sample

        Args:
            index (int): index of the entries

        Returns:
            dict: sample::

            {
                "input_a": feat_a,
                "input_b": feat_b,
                "output": [label]
            }
        """
        audio_data_a, speaker_a, audio_data_b, speaker_b, label = self.entries[index]
        feat_a = self.audio_featurizer(audio_data_a)
        feat_a = self.feature_normalizer(feat_a, speaker_a)
        feat_b = self.audio_featurizer(audio_data_b)
        feat_b = self.feature_normalizer(feat_b, speaker_b)
        return {
            "input_a": feat_a,
            "input_b": feat_b,
            "output": [label]
        }

    @property
    def sample_type(self):
        """:obj:`@property`

        Returns:
            dict: sample_type of the dataset::

            {
                "input_a": tf.float32,
                "input_b": tf.float32,
                "output": tf.int32
            }
        """
        return {
            "input_a": tf.float32,
            "input_b": tf.float32,
            "output": tf.int32
        }

    @property
    def sample_shape(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "input_a": tf.TensorShape([None, dim, nc]),
                "input_b": tf.TensorShape([None, dim, nc]),
                "output": tf.TensorShape([None])
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return {
            "input_a": tf.TensorShape([None, dim, nc]),
            "input_b": tf.TensorShape([None, dim, nc]),
            "output": tf.TensorShape([None])
        }

    @property
    def sample_signature(self):
        """:obj:`@property`

        Returns:
            dict: sample_signature of the dataset::

            {
                "input_a": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "input_b":tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return (
            {
                "input_a": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "input_b":tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            },
        )
