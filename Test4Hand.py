from mingus.midi import fluidsynth
import mingus.core.notes as notes
from mingus.containers import Note, Instrument
from mingus.containers import MidiInstrument
from mingus.containers import NoteContainer
from mingus.containers import Track


def coordinate_to_Note(X, Y):
    # set the column width for calculate
    column_width = 50
    note_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # x y correspond to coordinate
    x_int = int(int(X) / column_width)
    y_int = int(int(Y) / column_width)

    n = Note(note_list[x_int], y_int)

    return n

def loop_coordinate():
    t = Track()
    X = input("X:")
    Y = input("Y:")
    note_play = coordinate_to_Note(X, Y)
    t.add_notes(note_play)

    return t


if __name__ == '__main__':
    fluidsynth.init("GeneralUser_GS_1.471/GeneralUser_GS_v1.471.sf2")
    fluidsynth.set_instrument(1, 50)
    begin = input("e/c:(enter 'e' as 'end' and 'c' as 'continue'):")

    while begin != 'e':
        t = loop_coordinate()
        fluidsynth.play_Track(t, 1, 60)
        begin = input("e/c:(enter 'e' as 'end' and 'c' as 'continue'):")

    print('bye')

