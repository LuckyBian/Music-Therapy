import mingus
from mingus.midi import fluidsynth

fluidsynth.init("test.sf2")

while True:
    fluidsynth.play_Note(Note("C-5"))

