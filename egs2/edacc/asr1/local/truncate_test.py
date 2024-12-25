import os
import sys
import shutil
import wave


def truncate_test_set(test_dir, utterance_splits):
    """
    Truncate specified utterances in a Kaldi test set directory and reorganize the folder.

    Args:
        test_dir (str): Path to the Kaldi test set directory.
        utterance_splits (dict): A dictionary where keys are original utterance IDs, and values are lists of tuples,
                                 each containing (new_utterance_id, start_time, end_time).

    Returns:
        None
    """
    # Paths to Kaldi files
    wav_scp_path = os.path.join(test_dir, "wav.scp")
    text_path = os.path.join(test_dir, "text")
    utt2spk_path = os.path.join(test_dir, "utt2spk")
    segment_path = os.path.join(test_dir, "segments")

    # Temporary storage for new data
    new_text = []
    new_utt2spk = []
    new_segment = []
    # Check existence of Kaldi files
    if (
        not os.path.exists(text_path)
        or not os.path.exists(utt2spk_path)
        or not os.path.exists(segment_path)
    ):
        print("Error: Kaldi files not found in the specified directory.")
        sys.exit(1)

    # update variables

    with open(text_path, "r") as text_file_input:
        for line in text_file_input:
            utter, _ = line.strip().split(maxsplit=1)
            if utter in utterance_splits:
                for new_utter, _, _, new_txt in utterance_splits[utter]:
                    new_text.append(f"{new_utter} {new_txt}\n")
            else:
                new_text.append(line)

    with open(utt2spk_path, "r") as utt2spk_file_input:
        for line in utt2spk_file_input:
            utter, spk = line.strip().split()
            if utter in utterance_splits:
                for new_utter, _, _, _ in utterance_splits[utter]:
                    new_utt2spk.append(f"{new_utter} {spk}\n")
            else:
                new_utt2spk.append(line)

    with open(segment_path, "r") as segment_file_input:
        for line in segment_file_input:
            utter, id, start, end = line.strip().split()
            if utter in utterance_splits:
                for new_utter, new_start, new_end, _ in utterance_splits[utter]:
                    new_segment.append(f"{new_utter} {id} {new_start} {new_end}\n")
            else:
                new_segment.append(line)

    # Write Kaldi files
    with open(text_path, "w") as text_file:
        text_file.writelines(new_text)

    with open(utt2spk_path, "w") as utt2spk_file:
        utt2spk_file.writelines(new_utt2spk)

    with open(segment_path, "w") as segment_file:
        segment_file.writelines(new_segment)

    # print("Test set truncation and reorganization completed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python truncate_test.py [test set directory]")
        sys.exit(1)

    test_dir = sys.argv[1]

    # Example input: specify utterance splits
    utterance_splits = {
        "EDACC-C34-A-EDACC-C34-000000008": [
            (
                "EDACC-C34-A-EDACC-C34-000000008_0",
                0.00,
                130.12,
                "okay that's uh that's quite a good collection quite versatile collection i would say and you were able to mention some um good books as well other than you know movies and cartoons because if i you know look back in time and uh talk with reference to the kind of books that i've read so those were mostly uh textbooks you know that were part of the syllabus and i would even you know my family used to push me that uh you should develop this reading habit and you uh you should be fond of reading trying out some new story books and reading new literature new genres or even poems but uh you know every time i would get an opportunity to have some free time in which i could do something of like my own hobbies so i would always prefer to either play a game uh be it virtual where i would simply find some friends to go out and you know play cricket or football so that was the way in which i would you know spend my time other than that so you know that slot of reading some sort of book remained uh empty for for quite a number of years until the time came when i started to realize that there is quite a hype and you know fame of the harry potter saga i would say or the harry potter series you know everyone was just a diehard fan of harry potter be it the sorcerer's stone or the chamber of secrets or the prisoner of azkaban the goblet of fire and it goes on and on you already know i guess and i believe that you're also a big fan of harry potter so uh i i do remember i was i i was not even a teenager even younger than that so i guess uh one of my mom's friends simply came up and it was i guess my birthday and uh she gave me two things uh there was a board game and we should call it a board game which enhances our vocabulary so it was scrabbles so uh",
            ),
            (
                "EDACC-C34-A-EDACC-C34-000000008_1",
                130.12,
                266.52,
                "at that time actually honestly i was unable to figure it out what is it about and how are we supposed to play it uh and the other thing was that uh you know it was a cam print of the very first movie of harry potter harry potter and the sorcerer's stone so i was just wondering i still said okay yeah uh like she has gifted me uh this movie so i have to try it it will be something special so it it was either because of the cam version uh that i was unable to actually see and clearly observe what those characters were actually doing or maybe like i didn't find the plot of the movie very much interesting because i haven't read by that time i haven't read any of the books so that's why i was unable to figure it to figure it out like what is actually going on and what is happening i guess it was the next day or the a day after the next day day after tomorrow i mean so when i just you know i was just going downstairs to play with friends so i met the because she was a neighbor uh you know the mom's uh mom's friends that i was just mentioning who gifted me with harry potter so i just came across her and she just asked me so did you like the gift how was the movie and i was like okay yeah and uh it was okay it was nice but she recognized and she you know realized from my facial expressions that i kind of didn't like that movie and she said didn't you like the movie i just asked uh you know the shopkeeper and he was also saying that this is a very famous movie and people are going crazy about it i said okay fine yeah she uh it probably would be a very famous movie well i found it okay okay it was good but yeah it was you know um i felt a little shy at that time because uh it didn't look nice you know but still i tried to cover it up so that's how the story begins like i started witnessing you know observing so many friends in my neighborhood talking about harry potter either the book or the movie and you know i would see them you know exchanging cds or books so i started to you know wonder what's what's in it what's so crazy what's so special about this movie i guess i either have to",
            ),
            (
                "EDACC-C34-A-EDACC-C34-000000008_2",
                266.62,
                419.24,
                "you know watch the movie clearly or i have to just read one of these books of course uh it's quite obvious that you have to start from the very first one so then i asked one of my friend you know that uh i need the harry potter and the sorcerer's stone if you have it so one of my friends simply handed it over and i started reading it so yeah gradually you know you tend to indulge within the book very gradually and certainly when you start to feel like you're also one of those characters or at least an observer of everything that is happening in the book so then with the passage of time i started to develop the interest and it got i got so much into it and i just had the wish of reading the second one and the third one and then it it went on and on so uh i will simply you know uh put it at the top of the list of those books that i have read you know for fun otherwise i would simply say all the other books belong to the syllabus you know uh study english literature uh so i will say that all the books all the works uh that i've read of like william shakespeare or wordsworth or milton are all the one that were part of the syllabus one way or the other they are connected with the syllabus the one that i've simply chosen with my own will was harry potter but still you would say uh my my data my reading data is quite limited yeah i guess i have to read it but once i'm done with my phd so then i have this wish of you know going through a classic literature as well as the modern and the postmodern so i guess i have taken quite a lot of time but you know uh when we have to have a like a gradual or i would say casual conversation so that is how it happened natural conversation just you know uh happens to decide its own path so talking about uh the cartoons uh i guess i remember i used to watch from fiev to seven p m had tom jerry kids then jonny quest then the mask and lastly captain planet so these were the famous cartoons of the nineties and i i really enjoyed watching them quite a lot and after a couple of years then you know we had this dish through which i was able to you know watch cartoon network all the time and then",
            ),
            (
                "EDACC-C34-A-EDACC-C34-000000008_3",
                419.24,
                535.96,
                "i started watching like all the cartoons i guess i would i would remember the names of even all those cartoons the ones that you also mentioned let's say powerpuff girls i have watched it dexter's laboratory so i've watched that one too then wacky races and the list goes on and on but but the point is yeah these were some of the favorite i have i do have watched flints flintstones and jetsons those were also quite good so i really enjoyed those one also talking about the movies movies movies movies uh yeah terminator has always you know you know engaged me quite a lot though at that time when like when i was a kid i couldn't understand what was actually happening or why is this t one thousand you know turning into something we could how was he able to change shapes and how was even arnold schwarzenegger alive after getting so many bullets on his face and body so so many things happening i couldn't get it but i would simply enjoy the action and no matter how many times you're going to play that movie i would always love it and still love it now talking about my age of course now i've understood all the plot i know what was happening what are cyborgs and uh what is judgment day and what was actually happening so i did try uh you know uh i won't say try i know i do enjoy everything and i always wish you know somehow or the other that some producers or directors are going to come up with a new installment or a new uh version of terminator also and i'll be excited to see that movie in a cinema too so i will simply put that movie at the top i always loved it i loved watching it and i would be happy to see it again whenever i get the opportunity well i guess i have <laugh> i guess i have taken fifteen minutes to answer this question i am not sure or i guess",
            ),
        ],
        "EDACC-C34-A-EDACC-C34-000000018": [
            (
                "EDACC-C34-A-EDACC-C34-000000018_0",
                0.00,
                117.52,
                "well that is nice um i guess uh if we talk about the hobbies i tend to answer that uh question in a in such a way that that answer tends to answer this question too because uh watching tv i will say alongside uh involving myself in different sports is like the kind of hobby that i want to do or i've i've loved uh spending time in let's say playing cricket or uh football uh were two of my great hobbies in which i have spent some quality time that i always love to think about and other than that when i would you know return back from school after so many classes so i would just simply put my bag uh in a room and i would just simply turn on the tv tune into cartoon network and simply start watching those cartoons again and again though i would know what is going to happen next in the scene like i have watched those cartoons so many times but still i would simply love every time though the same episode would uh be repeated or let's say would be played again by a uh um broadcast or transmitted again from that channel i would simply love to watch it i would enjoy it in the same way as though it's being you know played for the very first time so i loved uh watching cartoons of all types i remember there was one there were so many very funny kind of cartoons too that were you know tuned or that were broadcast or transmitted from this cartoon network channel like cow and chicken was one of those cartoons like i i still don't understand what's the point of that cartoon it was so funny i couldn't even get what those two characters were doing and they would simply love themselves as to their siblings",
            ),
            (
                "EDACC-C34-A-EDACC-C34-000000018_1",
                118.94,
                237.88,
                "so uh then there was i guess another one i am weasel that was another uh cartoon so they were so many but i think with the passage of time yeah i i started to find some good series as well when i i entered within my teens so i was happy to you know uh find some great series cartoon series such as the justice league that that was a very great series that i really enjoyed watching during my uh teenage then the he man and the masters of the universe that was another cartoon that i really loved watching so these were the good ones uh that i do remember and i'm very happy other than that i guess when i was uh in my school um you know in the beginning teenage so i watched the toonami series in which there were two cartoons i guess so one was um beyblade beyblade series v force g revolution and then there were the digimons and the pokémon series so these were the few of the cartoons yeah i would say like a hobby i would watch these cartoons and i really enjoyed my time then the second part of the question is do you still uh you know uh involve yourself in some hobby so i would say as i mentioned uh you know uh being active being physical is something that i like to do to keep myself fit healthy and active so i have been hiking during my childhood you know to simply lose weight because i have remained quite obese during my i had this you call it childhood fat uh so you know uh i have spent quite a lot of time um in in hiking so to lose weight other than that i would say that i have played so many games now if i talk about uh",
            ),
            (
                "EDACC-C34-A-EDACC-C34-000000018_2",
                238.70,
                356.60,
                "the time that i simply spend so that is basically tennis tennis is the game that i really love to play i have been watching uh tennis since two thousand seven eight when i started to see federer and the way he simply plays with a racket as though it feels like as though there is a wand in his hand so he had so many nemeses like nadal and djokovic so they gave him a very tough time but still he managed to win twenty grand slams i'm not saying he's the greatest he's like one of the greatest player of all times there are two other nadal and djokovic so they're also doing great let's say nadal has like won his twenty second uh grand slam singles title uh recently by winning the roland garros title so i would simply say yeah these three are the biggest players of all times making records the un unbeatable records i would say so i have the same kind of fire and i do enjoy playing tennis with my friends in the evening or sometimes before going to the university early in the morning so that is one of the hobbies that i you know love spending time in uh and with that i will say uh i guess uh we got a notification also that um our meeting will close within ten minutes time so because i don't have this upgraded version of zoom so i guess it is time to uh you know sum it all up we did have a very great conversation and um i guess it will be quite useful for research and analysis so thank you very much f c two six p two it was a lovely time talking to you and it's uh goodbye from my side also f c two six p one signing out and p two signing out also thank you",
            ),
        ],
        "EDACC-C31-A-EDACC-C31_P2-000000153": [
            (
                "EDACC-C31-A-EDACC-C31_P2-000000153_0",
                0.00,
                153.80,
                "all story whole story you know whole story uh turns around that ship somehow that uh medical uh you know scientist she comes to the that ship and uh she meet with that person and she uh she uh somehow she encouraged that person to uh you know travel in different places and different island to find the cure but for cure she need uh the infected person blood and it's really hard uh it's really hard to get infected blood uh from you know from zombie so when they're talking then she she's talking to that commander that she need uh blood infected blood and then after that she tried to uh you know kill that person all all the people so in this conversation he uh somehow she convinced that person that commander and uh he said okay i will bring a cure for you as well uh i will bring uh you know infected uh blood for you so mm after that they uh normally they uh they're showing few days they just uh travel uh around uh you know travel in water because la everywhere in land everywhere is infection uh every person is infected so they uh they travel you can say they travel uh few days in water then after that they and the they come to near the land then uh as the commander promised to that person or you can say uh scientists then they come down and they bring some a few people uh few mili military people come down you can say some marine they come down with that with that girl to catch uh infected person because it's very hard and it's it's very scary to catch someone uh some kind of zombies so then after that they catch uh after you know after that they catch some kind of uh you can say a few people a few people they catch some few people but they don't know there's a lot of more over there to mm you know uh to infect there certainly the uh whole um",
            ),
            (
                "EDACC-C31-A-EDACC-C31_P2-000000153_1",
                153.80,
                269.24,
                "you know the whole area come out and they see few people they ran there behind them uh so they ran away and few people that died over there are infected over there and they again when then after that they went back and uh when they went back they they checked theirself because if someone bites or someone you know if someone bites infected per person or you can say well zombie bite some human being and so they will the they all will be infected and also there is no cure for that uh for that uh thing so for in first try when they uh try to catch a zombie uh the commander lost their you can say five to ten people so when they went back they are they are back to the shape and they again they tra they travel in the water they spend few days over there and again she uh you can say a scientist girl tried to convince that person we uh we need a cure if we don't have uh you know we have a few days food left in our ship and uh we don't have uh much fuel to travel and we can't travel whole life in the ship so we have to find a cure to cure uh whole community or you can say our country or our earth and this drama they're showing like that so then after that again he uh he comma he commands and he says okay let's do this time so then after that they uh went there they when they travel a few days and after that they tried to come out near the land then after that they try again to mm uh go out and that was also uh it's really hard for them to do that this time as well",
            ),
        ],
        "EDACC-C31-A-EDACC-C31_P2-000000139": [
            (
                "EDACC-C31-A-EDACC-C31_P2-000000139_0",
                0.00,
                124.52,
                "time they were doing uh you know they are making different kind of cure to prevent the different disease in that movie they are showing they are doing you know different kind of and the and the main character is uh the main character in that movie is two the the first one is a girl she is uh uh bioche bio i think bio doctor biochemical doctor she always makes something to uh you know uh cure some kind of disease for human being or you can say uh for uh any kind of animals or you can say human beings as well she all the time she do uh try to make a cure for that (()) that but there is you know everybody you know that everywhere is a good people and there's a bad people as well in that mo in that movie they they are showing you know yeah in that movie they are the two type of people there there's a terrorist they uh they are making different kind of uh you can say uh what they call uh you know uh bioweapons they want they are making bioweapons they use their knowledge for negative things in that movie they are uh uh they go to some kind of uh what they call uh island they have their island they went there they have a lot of money in the in this uh in this uh uh you know in this drama they're showing they have they have a lot of money then after that they buy uh island then they go there and they make it their lab over there and after that they make a team and they started you know negative thing uh they start to they started on different thing on but at first they start on different kind of animals like monkey and different kind of rats and then after that they started uh you know making negative things like zombies they start they make zombies for before first days they test on uh rats",
            ),
            (
                "EDACC-C31-A-EDACC-C31_P2-000000139_1",
                124.52,
                240.12,
                "then after that the monkey because monkeys are really near to human being so then after that they uh inject uh to a human being they inject that uh dose or you can say biochemical to that then after that that human that person was a security guard and that turned into zombies and they just grab them in you know uh different kind of uh their weapons and they grab them they want to use them for negative negative things but all at at that times there is an incident incidents happened in that lab and suddenly the zombie they uh come out their cage and they you know divide different uh all the people over there which are available in lab and somehow uh you know over there that uh on that island every time when there is a holiday every time the tourists come there to get island uh to visit their uh visit there and spend their time over there but they don't know there is a you know hidden lab over there and people are wrong people are doing experiment over there so this time you know in that drama they are changing this time there is uh few people are you know we can say it's uh uh you know big boat come there uh and when they and they just come there and spend their holidays over there and suddenly in the evening time that uh infected people come out from the lab and they bite all the people and everyone is turned human being to zombie uh step by step they are showing that the zombies are increasing step by step okay hello can you hear me",
            ),
        ],
        "EDACC-C34-A-EDACC-C34-000000006": [
            (
                "EDACC-C34-A-EDACC-C34-000000006_0",
                0.00,
                115.88,
                "okay uh that is awesome uh very nice uh and now if i tend to answer the same question and looking back in time uh the kind of games that i've played are quite simple uh snakes and ladders and ludo so these were the ones in which you know it's all based on luck if the dice comes to six yeah uh i would say woohoo i can you know come out of the place and move around and to reach at my destination so that would give me all the happiness that i was looking for so it was either snakes and ladders or ludo and yeah from cards uh there was a game that i remember uh that i learned from a couple of friends it was yeah snap in which you know if uh two cards of the same shape appear you know placed by two different players so the one that sees that yeah the two of them are of the same shape and simply put their hand very quickly on those cards and say snap so they are going to get it and the one who gets you know the majority of the card simply wins the game so uh that was the game that i remember i did learn uh a few games you know this year or uh the previous year when we were going through the pandemic so but you know i'm not able to remember them quite well uh but the ones that i learned are you know played in childhood so those were the good ones and then that the ones that i still remember so they were kind of funny too so other than that i would say i was more into the physical kind of games the outdoor games uh that we can call as the the extracurricular activities in which i would simply go out and play cricket with neighborhood friends and playing football uh badminton ",
            ),
            (
                "EDACC-C34-A-EDACC-C34-000000006_1",
                116.14,
                199.24,
                "so these were the kinds of games that i have played quite a lot and yeah uh there was a place in which some you know uh the ar arcade games were also available you know they were at a walking distance from our res residence so sometimes like daily at the evening hours and sometime during weekends so me and my family would simply go out they would simply have something to eat and i would go there you know to play the arcade games so you know games like street fighter king of fighter so these these were the kinds of games that i remember to be playing and i kind of enjoyed them a lot so and i would simply call them to be the ones which were like one of my favorite ones because i used to play them regularly and would simply enjoy uh each and every time in the same way as i am playing them for the very first time so that was like kind of fun so i hope that uh we both have coded up quite well if we talk about the kinds of games and the ones that we remember so if i ask you uh the second question would be that did you have any or i would simply say did you have a favorite book or any film or some tv show or any cartoon as a child that you would love to watch read so what was it about",
            ),
        ],
        "EDACC-C31-A-EDACC-C31_P1-000000053": [
            (
                "EDACC-C31-A-EDACC-C31_P1-000000053_0",
                0.00,
                92.44,
                "yeah ev every week they show around yeah you can say forty to forty five minutes of drama but now uh uh when i see the last episode uh they were showing that uh he's planning to break that prison he he's architect and he is planning to uh break that prison uh in uh in that drama they are showing that he you know he ta tattoo uh uh you know the blueprint of that prison the blueprint of that prison in his body and whole he tattoo whole body and it shows like uh you can say different kind of tattoos and you know but actually that is a blueprint of that prison okay he printed everything on uh himself then after that he planned to uh he planned to go that also he planned to go to the prison as well well then after that uh first of all he uh did uh you can say tattoo himself with the blueprint of that prison then after that he uh you know he got the gun and put in his pocket and he went to the what they call uh he he went to the bank and over there he fired uh he fired in the air then uh the security guard catch him then they send him to that prison the same prison they sent him where his brother is then over there ",
            ),
            (
                "EDACC-C31-A-EDACC-C31_P1-000000053_1",
                92.18,
                198.84,
                "uh over there he started to uh see oh here and there and watch try to you know want to uh he tried to see the weak points of the prison and also he have the um you know full map or you can say blueprint on his body over there in prison every every time whenever when that was his first day and every day you know mm there's a lot of prisoner over there he see over there in prison they are showing in prison there's different parties or you can say different people there different gangs in the prison as well some you know some chinese some uh you know mm red indian like uh red indian and some others are different parties he was uh alone and he he is kind of quiet person and he you know observed everything they are showing that he observed everything what's going on over there and he look around and he sends everything and he to every time he you know he moves out and he pla when they you know uh they can get them out to play the prisoner even police allow him to go for the lunch dinner breakfast or outside and they also every day they do they work as well when they go outside and uh when they go outside they will uh do different kind of work and over there he always uh every time he go out and he see different things and he try to find out uh to break the prison that's the last they are showing in uh that uh you know in the last video and also he's yeah",
            ),
        ],
        "EDACC-C31-A-EDACC-C31_P2-000000161": [
            (
                "EDACC-C31-A-EDACC-C31_P2-000000161_0",
                0.00,
                104.44,
                "some zombie and they are they are trapped in uh you know in a building there's uh all around the building there's a lot of zombies but they catched only single person uh in the uh building room and also they are uh they can't go out to the you know water or their ship so they also catch this time after that the story uh the story comes around it you can say two or three epito episode uh are gone over there in in in that building they they the that was a shopping mall or you can say these kind of things and also they uh over there there's a lot of food available they spent few days and uh she tried to you know she tried to get uh blood of that one and uh finally she tried to get the blood and it's really hard for her um because she is very scared from the zombies and this kind of infected people then after that she got uh somehow uh captain helped her and all the military person or marines help helped helped her to get the blood so finally she got blood from she got blood from that zombie and she uh she's trying to uh then after that she is started experimenting on that blood okay so somehow they spend few days over there and finally they you know when were there the zombies knows about them and they attack that building and also they attack that building uh they attack that building and somehow they get out of that building through like you can say uh ",
            ),
            (
                "EDACC-C31-A-EDACC-C31_P2-000000161_1",
                104.44,
                185.00,
                "i think it's electric lines or electrical lines over there so they come out from there and they went and they you know they ran away from there and some uh it's very hard for them they are showing it's very hard for them they travel through bike some some travel through through bikes and some through a car and this kind of stuff and they don't have even they uh they ended with uh they don't have bullets now but somehow they reach their um uh you know their place to uh you can say uh beach area where the ship is uh available so then after that uh there's um uh many people they went there there's a team but only three to five people left they get back to the ship and then after that they tried experimenting but need she now she need uh you know cure to make a cure and they have to you know they have to travel to different uh you can say uh different island for for the what they call a different kind of trees or different kind of things they need to make a cure are you enjoying are you feel boring you don't have a question like that <laugh>",
            ),
        ],
        "EDACC-C57-A-EDACC-C57_P2-000000134": [
            (
                "EDACC-C57-A-EDACC-C57_P2-000000134_0",
                0.00,
                106.20,
                "yeah but maybe you know i considering what this worker said to me i just uh first of all uh there are two things first of all i recall that a situation like this happened very similar when i went to the girls you remember and i went there very upset at the time i was very upset it was the same situation i was very troubled about something that they they did okay and i went there and i burst like out everything that i thought so that they reacted as a resent with resentment and this damage could not be uh like like uh recovered in any way so there was a lack of trust all of a sudden and they didn't you know and they were so shocked because i ga i went there and i was out of my uh my you know my horses this time happened the same thing i i sent that picture because i was upset and i want them to admit that you know to validate my upsetting but their re their reaction okay in that sense with the girls i honestly okay i admit again the mistake of acting as you say not in a rational way or in a calm way upon emotions sudden emotions but i think i was right on what i was saying i could have said the same thing in a much calmer way same thing with them ",
            ),
            (
                "EDACC-C57-A-EDACC-C57_P2-000000134_1",
                106.70,
                172.28,
                "i could have delivered the message in in a better and more diplomatic way okay in both in do in both cases i certainly made a mistake in management okay but in this case of the workers i ask myself considering afterwards if maybe i don't recognize that maybe they are doing an effort beyond you know what i'm paying them for i i have the doubt i'm not completely sure because consider i have been waiting for three months for them to finish this work i wouldn't call it such an effort you know but certainly the boss the mm vladimir certainly is has been very elastic with me very flexible very understanding et cetera i don't know i don't know anyway nothing justifies an humiliation like the one of receiving a video like that saying that your job is bad no so it was an abuse from my part in",
            ),
        ],
    }

    truncate_test_set(test_dir, utterance_splits)
