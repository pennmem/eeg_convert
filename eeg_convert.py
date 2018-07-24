from argparse import ArgumentParser
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import h5py
import numpy as np
import tables  # enables BLOSC
from tqdm import tqdm

from cmlreaders import CMLReader, PathFinder
from cmlreaders.util import get_root_dir


class EEGConverter(object):
    def __init__(self, subject, experiment, session, outdir, rootdir=None):
        self.rootdir = get_root_dir(rootdir)
        self.outdir = Path(self.rootdir).joinpath(outdir)
        self.reader = CMLReader(subject, experiment, session, rootdir=rootdir)

        sources_filename = self.reader.path_finder.find("sources")
        with open(sources_filename, "r") as infile:
            self.sources = json.load(infile)

        self.eeg_files = [
            sorted(
                Path(sources_filename)
                .parent
                .joinpath("noreref")
                .glob(info["name"] + "*")
            ) for info in self.sources.values()
        ]

    @property
    def num_channels(self):
        return len(self.eeg_files[0])

    @property
    def dtype(self):
        for sources in self.sources.values():
            return sources["data_format"]

    def labels_as_array(self) -> np.ndarray:
        contacts = self.reader.load("contacts")
        strlen = contacts.label.str.len().max()
        labels = contacts.label.values.astype(f"|S{strlen}")
        return labels

    def to_hdf5(self, filename="eeg_timeseries.h5"):
        """Convert to HDF5."""
        outpath = self.outdir.joinpath(filename)

        with h5py.File(outpath, "w") as hfile:
            labels = self.labels_as_array()
            hfile.create_dataset("labels", data=labels, chunks=True,
                                 compression="gzip")
            start_timestamps = []

            for dset_num, info in tqdm(enumerate(self.sources.values())):
                dtype = info["data_format"]
                sample_rate = info["sample_rate"]
                start_timestamps.append(info["start_time_ms"] / 1000.)

                files = self.eeg_files[dset_num]
                num_channels = len(files)
                dset = None

                for ch, filename in tqdm(enumerate(files)):
                    data = np.fromfile(str(filename), dtype=dtype)

                    if dset is None:
                        shape = (len(self.sources), num_channels, len(data))
                        dset = hfile.create_dataset(
                            "eeg", shape,
                            dtype=info["data_format"],
                            # chunks=True,
                            # compression=(32001 if compress else None),
                            # compression_opts=9,
                            # shuffle=True,
                        )

                    dset[dset_num, ch] = data

            start_dset = hfile.create_dataset("start_time", data=start_timestamps)
            start_dset.attrs["desc"] = b"unix timestamp of session start"

            hfile.create_dataset("sample_rate", data=sample_rate)

    def to_npz(self, filename="eeg_timeseries.npy"):
        """Convert to numpy's format."""
        outpath = self.outdir.joinpath(filename)

        arrays = {"labels": self.labels_as_array()}
        eeg = None

        for dset_num, info in tqdm(enumerate(self.sources.values())):
            files = self.eeg_files[dset_num]

            for ch, path in tqdm(enumerate(files)):
                with path.open() as f:
                    data = np.fromfile(f, dtype=self.dtype)

                if eeg is None:
                    shape = (len(self.sources), self.num_channels, data.shape[0])
                    eeg = np.empty(shape, dtype=self.dtype)

                eeg[dset_num, ch] = data

        arrays["eeg"] = eeg
        np.save(outpath, eeg, allow_pickle=False)
        # np.savez(outpath, **arrays)


def main():
    parser = ArgumentParser()
    parser.add_argument("--subject", "-s", type=str, default="R1111M")
    parser.add_argument("--experiment", "-x", type=str, default="FR1")
    parser.add_argument("--session", "-n", type=int, default=0)
    parser.add_argument("--outdir", "-o", type=str, default="scratch/depalati")

    args = parser.parse_args()

    converter = EEGConverter(args.subject, args.experiment, args.session, args.outdir)
    # converter.to_npz()
    converter.to_hdf5("no_chunks.h5")


if __name__ == "__main__":
    main()
