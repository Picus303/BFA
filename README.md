# BFA Forced-Aligner (Text/Phoneme/Audio Alignment)

A CLI Python tool for text/audio alignment at word and phoneme level.<br />
It supports both **textual and phonetic input** using either the [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) and [Misaki](https://github.com/hexgrad/misaki) phonesets.<br />
The integrated G2P model supports both **british and american English**.<br />
The final alignments are output in [**TextGrid**](https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html) format.

It's based on a RNN-T model (CNN/LSTM encoder + Transformer decoder) and was trained on 460 hours of audio from the [LibriSpeech dataset](https://www.openslr.org/12).<br />
The current architecture only supports audio clips up to about 17.5 seconds (see [contributions](#Contributions)).

**No GPU is required to run this tool**, but a CPU with lots of cores can help.


## Installation

```bash
pip install BFA
```

**Requires Python ≥ 3.12**<br />


## Usage (CLI)

To align a corpus, two directories are expected:
- One that contains all your audio files (**.wav**, **.mp3**, **.flac** and **.pcm** files only)
- One that contains all your annotations (**.txt** and **.lab** files only)

You can find examples of such files in the [example](./example) directory of this repository.
A recursive search will be used so **the only constraint is that both directories uses the same structure**. If you use the same directory for both, then your .wav and .lab pairs should be in the same sub-directory.

```bash
bfa align \
  --audio-dir /path/to/audio_dir \
  --text-dir /path/to/text_dir \
  [--out-dir /path/to/out_dir] \
  [--dtype {words, phonemes}] \
  [--ptype {IPA, Misaki}] \
  [--language {EN-GB, EN-US}] \
  [--n-jobs N] \
  [--ignore-ram-usage] \
  [--config-path /path/to/config_file] \
```


## Performances

Aligning the **460 hours** of audio of the LibriSpeech dataset took **2h30 (realtime factor: x184)** on a **8 cores / 16 threads CPU**. Realtime factor on one core: x11.5.<br />
**1.5Go of RAM per thread are required** (here, 24Go for 16 threads). By default, BFA will check your total RAM before starting jobs.<br />
It **successfully aligned** more than **99%** of the files.<br />


## To Do:

- Test IPA ptype
- Test Word dtype


## Contributions

All contributions are welcomed but my main goal is the following:

Currently, the main limitation of this tool is it's context length (about 17.5 seconds) but RNN-T models can use a streaming implementation and this way handle files of arbitrary sizes.
This would requires making the model causal (currently it isn't, in order to maximize accuracy) and write an inference function that can handle this method.

It would also be interesting to support .TextGrid files for annotations (input).


## Licence

This project is licensed under the MIT License. See the LICENSE file for details.