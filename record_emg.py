import sys
import os
import curses
import json
import numpy as np
import soundfile as sf

from record_data import Recorder

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_string('output_directory', './from', 'where to save outputs')
flags.DEFINE_string('word_to_record', 'test', 'word to record')
flags.mark_flag_as_required('output_directory')

def save_data(word, emg, audio):
    emg_file = os.path.join(FLAGS.output_directory, f'{word}.npy')
    audio_file = os.path.join(FLAGS.output_directory, f'{word}.flac')

    np.save(emg_file, emg)
    sf.write(audio_file, audio, 16000)  # Assuming sample rate is 16000 Hz

    info_file = os.path.join(FLAGS.output_directory, f'{word}.json')
    with open(info_file, 'w') as f:
        json.dump({'word': word}, f)

def main(stdscr):
    os.makedirs(FLAGS.output_directory, exist_ok=True)

    curses.curs_set(False)
    stdscr.nodelay(True)

    recording = True  # Start recording immediately

    with Recorder(debug=FLAGS.debug) as r:
        stdscr.clear()
        stdscr.addstr(0, 0, "Recording... Press 'q' to stop.")
        stdscr.refresh()

        while recording:
            r.update()
            c = stdscr.getch()
            if c == ord('q'):
                recording = False
                emg, audio, _, _ = r.get_data()
                save_data(FLAGS.word_to_record, emg, audio)
                break

FLAGS(sys.argv)
curses.wrapper(main)

