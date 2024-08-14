from mingus.midi import fluidsynth
import mingus.core.notes as notes
from mingus.containers import Note, Instrument
from mingus.containers import MidiInstrument
from mingus.containers import NoteContainer
from mingus.containers import Track

#fluidsynth.set_instrument(1, 14)
t = Track()
i = MidiInstrument()
i.midi_instr = 80
t = Track(i)
fluidsynth.init("test.sf2")
fluidsynth.set_instrument(1, 14)

Note_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
octave_list = [*range(-1, 9, 1)]
# a = Note("A", 5)
n = NoteContainer([Note(Note_list[2], 8), Note(Note_list[1], 5), Note(Note_list[5], 5)])
n2 = NoteContainer(["B-3", "C-5", "G#-5"])
n3 = NoteContainer(["A#-3", "B-5", "G-5"])
#fluidsynth.play_Note(Note("C-5"))
t.add_notes(n)
t.add_notes(n2)
t.add_notes(n3)

fluidsynth.play_Track(t,1)
