from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import tables  # enables BLOSC
from tqdm import tqdm

from cmlreaders import CMLReader, PathFinder
from cmlreaders.util import get_root_dir


def split_to_hdf5(subject: str, experiment: str, session: int,
                  outpath: Union[str, Path], compress: bool = True,
                  rootdir: Optional[str] = None):
    """Convert "split" EEG format to HDF5.

    :param subject:
    :param experiment:
    :param session:
    :param outpath: location to write the HDF5 file to
    :param compress: use compression when True
    :param rootdir: path to rhino root

    """
    rootdir = get_root_dir(rootdir)
    reader = CMLReader(subject, experiment, session, rootdir=rootdir)
    finder = reader.path_finder
    sources_filename = finder.find("sources")

    with open(sources_filename, "r") as infile:
        sources = json.load(infile)

    outpath = Path(rootdir).joinpath(outpath, "eeg_timeseries.h5")

    with h5py.File(outpath, "w") as hfile:
        contacts = reader.load("contacts")
        strlen = contacts.label.str.len().max()
        labels = contacts.label.values.astype(f"|S{strlen}")
        hfile.create_dataset("labels", data=labels, chunks=True, compression="gzip")
        start_timestamps = []

        for dset_num, info in tqdm(enumerate(sources.values())):
            dtype = info["data_format"]
            sample_rate = info["sample_rate"]
            start_timestamps.append(info["start_time_ms"] / 1000.)

            files = sorted(
                Path(sources_filename).parent
                                      .joinpath("noreref")
                                      .glob(info["name"] + "*")
            )

            num_channels = len(files)
            dset = None

            for ch, filename in tqdm(enumerate(files)):
                data = np.fromfile(str(filename), dtype=dtype)

                if dset is None:
                    shape = (len(sources), num_channels, len(data))
                    dset = hfile.create_dataset(
                        "eeg", shape,
                        dtype=info["data_format"],
                        chunks=True,
                        compression=(32001 if compress else None),
                        # compression_opts=9,
                        # shuffle=True,
                    )

                dset[dset_num, ch] = data

        start_dset = hfile.create_dataset("start_time", data=start_timestamps)
        start_dset.attrs["desc"] = b"unix timestamp of session start"

        hfile.create_dataset("sample_rate", data=sample_rate)


def main():
    parser = ArgumentParser()
    parser.add_argument("--subject", "-s", type=str, default="R1111M")
    parser.add_argument("--experiment", "-x", type=str, default="FR1")
    parser.add_argument("--session", "-n", type=int, default=0)
    parser.add_argument("--outpath", "-o", type=str, default="scratch/depalati")

    args = parser.parse_args()

    split_to_hdf5(args.subject, args.experiment, args.session, args.outpath,
                  compress=False)


if __name__ == "__main__":
    main()
