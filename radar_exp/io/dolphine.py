# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:04:35 2019

@author: Costas
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

class Recording:
    """Recording folder wrapper."""

    def __init__(self, directory):
        """
        Args:
            directory (string): Path to the folder that contains the recoded data.
        """
        if not os.path.isdir(directory):
            raise Exception('Folder not found!')
        self.directory = directory
        self.git = False
        self.raw_filename = 'raw {}.dat'
        self.processed_filename = 'processed {}.dat'
        self.camera_filename = 'camera0 {}.jpg'
        print("Parsing {}".format(directory))
        self.settings, self.profiles, self.meta, self.warnings = self.parse_info()
        print("Done!")

    def __repr__(self):
        return "<Recording at {}>".format(self.directory)

    def __str__(self):
        return "<Recording at {}>".format(self.directory) \
                + self.settings.__str__()

    def __len__(self):
        return self.meta['frames saved'][0]

    def __getitem__(self, idx):
        image = self.image(idx)
        cube = self.cube(idx)
        return image, cube

    def parse_info(self):
        filename = os.path.join(self.directory, "Settings.csv")
        settings = pd.read_csv(filename, nrows=1)
        profiles = pd.read_csv(filename, skiprows=2)
        filename = os.path.join(self.directory, "meta.csv")
        meta = pd.read_csv(filename)
        warnings = []
        sim_test = []
        for key, value in profiles.items():
            sim_test.append(len(np.unique(profiles[key])))
        if sum(sim_test[0:10]) != 10:
            warnings.append("Warning: profiles have different chirp settings!")
        d = {'txGain_1': [255, 0, 0], 'txGain_2': [0, 255, 0], 'txGain_3': [0, 0, 255],
             'txOn_1': [1, 0, 0], 'txOn_2': [0, 1, 0], 'txOn_3': [0, 0, 1]}
        df = pd.DataFrame(data=d)
        if not df.equals(profiles.iloc[:, 10:16]):
            warnings.append("Warning: profiles have wrong power settings!")
        return settings, profiles, meta, warnings

    def image(self, idx):
        filename = os.path.join(self.directory, self.camera_filename.format(idx))
        return Image.open(filename)

    def cube(self, idx):
        chirps = self.settings['chirps/tx'].item()
        samples = int(self.settings['all samples'].item() / self.profiles.iloc[0]['deci'])
        ant = self.settings['virtual antennas'].item()
        tx = self.settings['tx antennas'].item()
        rx = self.settings['rx antennas'].item()
        filename = os.path.join(self.directory, "raw " + str(idx) + ".dat")
        try:
            frame = np.fromfile(filename, dtype=np.uint16)
            frame = (frame / 2048.0) - 1.0
            frame = frame.reshape(rx, chirps, tx, samples)
            cube = np.empty((ant, chirps, samples), dtype=np.float64)
            for t in range(tx):
                for r in range(rx):
                    cube[t*rx+r] = frame[r, :, t]
        except FileNotFoundError:
            cube = np.ones((ant, chirps, samples), dtype=np.float64)
        return cube
