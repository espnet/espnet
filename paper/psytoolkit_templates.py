MOS_SECTION_TEMPLATE = """
page: begin

l: mos
t: info
q: Welcome to the listening experiment! It would be good to use earphones. For this section, please rate how natural the audio is from a scale of 1 (totally unnatural) to 5 (totally natural).

l: name
t: textline
q: Please input your name first :)
-

random: begin

{mos_questions}

random: end

l: end_of_mos
t: info
q: Congrats! Please click the button below to continue. (If you cannot click it, please check if you missed out something. All the bars should be green.

page: end
"""

MOS_QN_TEMPLATE = """
l: {wav_id}
a: {hosting_url}/{wav_path}
t: range
q:
- {{min=1,max=5,by=0.25,start=3,left=Totally Unnatural,right=Totally Natural}}
"""


PREF3_TEST_TEMPLATE = """
page: begin

l: pref3_test
t: info
q: Thank you for making it here! Just a bit more -- for this section, please rate which of the 3 audio samples has the most natural and least natural sound. If all sound the same, just select the same option for best/worst.

scale: abx
- 1st
- 2nd
- 3rd

{pref3_questions}

page: end
"""

PREF3_QN_TEMPLATE = """
l: {group_id}
a: {hosting_url}/{wav_paths[0]} {hosting_url}/{wav_paths[1]} {hosting_url}/{wav_paths[2]}
t: scale abx
- Most natural
- Least natural
"""
