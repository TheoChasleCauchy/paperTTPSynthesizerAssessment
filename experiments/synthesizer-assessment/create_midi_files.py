import mido
from mido import Message, MidiFile, MidiTrack
import os

def create_midi_files():
    for note in range(0, 128): 

        # Create MIDI file and track
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # MIDI settings
        tempo = mido.bpm2tempo(120)  # 120 BPM
        ticks_per_beat = mid.ticks_per_beat

        # Add tempo meta message
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))

        # Calculate duration
        seconds = 1.0
        beats_per_second = 120 / 60  # 2 beats per second at 120 BPM
        beats = seconds * beats_per_second
        ticks = int(beats * ticks_per_beat)

        # Note ON (MIDI note 64 = E4)
        track.append(Message('note_on', note=note, velocity=100, time=0))

        # Note OFF after 1 second
        track.append(Message('note_off', note=note, velocity=0, time=ticks))

        # Save MIDI file
        os.makedirs("data/TokenSynth/midi_files", exist_ok=True)
        mid.save(f"data/TokenSynth/midi_files/input_midi_{note}.mid")