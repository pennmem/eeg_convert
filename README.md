# EEG conversion tool

Convert split EEG to HDF5 for more efficient reading.

## File format description

The HDF5 file output is named `eeg_timeseries.h5`. It contains the following
entries:

* `eeg`: EEG timeseries data shaped like `(num_starts, num_samples, num_channels)`
  where `num_starts` is the number of times a given session was started
* `labels`: the labels associated with each EEG channel in the `eeg` dataset
* `sample_rate`: the sample rate in Hz
* `start_times`: Unix timestamps for the recording start times in seconds
