"""
midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)

# load tempo
tempos = midi_obj.tempo_changes
tempos.sort(key=lambda x: (x.time, x.tempo))
assert len(tempos) == 1
tempo = int(tempos[0].tempo + 0.5)

# load pitch & seg
tick_to_time = midi_obj.get_tick_to_time_mapping()
notes = midi_obj.instruments[0].notes
notes.sort(key=lambda x: (x.start, x.pitch))
for note in notes:
    st = tick_to_time[note["start"]]
    ed = tick_to_time[note["end"]]
    seg = [st, et]
    pitch = note.pitch

# rest note need to be added, pitch = 0
# rest note's seg can be calculated
# accordind to the breaks between the above notes sequence
"""
