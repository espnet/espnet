#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Roshan Sharma (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from xml.dom import minidom
import os
import urllib.request
import zipfile
import argparse 


class AMIDataExtractor:
    def __init__(self, ami_dir, in_file_ext='xml', out_file_ext='txt'):
        self.args = args
        self.in_file_ext = in_file_ext
        self.out_file_ext = out_file_ext

        self.ami_dir = os.path.join(ami_dir,'ami_public_manual_1.6.2')
        self.download_corpus()

    def download_corpus(self):
        """ Check if AMI Corpus exists and only download it if it doesn't

        :return: directory where AMI Corpus is located
        """
        download_link = 'http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip'

        if not os.path.exists(self.ami_dir):
            # 1. Ensure data directory exists
            if not os.path.exists(self.args.ami_xml_dir):
                os.mkdir(self.args.ami_xml_dir)
            # 2. Download AMI Corpus
            print("Downloading AMI Corpus to: {}".format(self.ami_dir))
            zipped_ami_filename = self.ami_dir + '.zip'
            urllib.request.urlretrieve(download_link, zipped_ami_filename)
            # 3. Unzip zip file
            zip_ref = zipfile.ZipFile(zipped_ami_filename, 'r')
            zip_ref.extractall(self.ami_dir)
            zip_ref.close()
            # 4. Delete zip file
            os.remove(zipped_ami_filename)
        else:
            print("AMI Corpus has already been downloaded in: {}".format(self.ami_dir))

    def get_corpus_directory(self):
        return self.ami_dir
    

    def get_transcript(self, conv_id):
        """ Extracts transcript for a given conversation ID

        :param conv_id: conversation ID
        :return: transcript
        """
        transcript = []
        word_speakers = []
        word_start_times = []
        word_end_times = []
        speaker_ids = [ x.split(".")[-3] for x in os.listdir(f"{self.ami_dir}/words/") if x.startswith(conv_id)]
        
        for speaker in speaker_ids:
            words_file = os.path.join(self.ami_dir,'words', f"{conv_id}.{speaker}.words.xml")
            xmldoc = minidom.parse(words_file)
            itemlist = xmldoc.getElementsByTagName('w')
            for s in itemlist:
                transcript.append(s.firstChild.data)
                word_speakers.append(speaker)
                word_start_times.append(s.getAttribute("starttime"))
                word_end_times.append(s.getAttribute("endtime"))
        
        sorted_data = sorted(zip(word_start_times,word_end_times,word_speakers,transcript),key=lambda x: x[0])
        trans = [x[-1] for x in sorted_data]
        return " ".join(trans)
    
    def get_abssum(self, conv_id):
        """ Extracts summary for a given conversation ID

        :param conv_id: conversation ID
        :return: summary
        """
        
        summ_file = os.path.join(self.ami_dir,'abstractive', f"{conv_id}.abssumm.xml")
        summary = ''
        try:
            xmldoc = minidom.parse(summ_file)
            abslist = xmldoc.getElementsByTagName('abstract')
            items_sentences = abslist[0].getElementsByTagName('sentence')
            for item in items_sentences:
                summary += item.firstChild.data + ' '
        except:
            print(f"Could not find summary for {conv_id}")
        
        actions = ''
        try:
            actlist = xmldoc.getElementsByTagName('actions')
            items_sentences = actlist[0].getElementsByTagName('sentence')
            for item in items_sentences:
                actions += item.firstChild.data + ' '
        except:
            print(f"Could not find actions for {conv_id}")
        
        decisions = ''
        try:
            declist = xmldoc.getElementsByTagName('decisions')
            items_sentences = declist[0].getElementsByTagName('sentence')
            for item in items_sentences:
                decisions += item.firstChild.data + ' '
        except:
            print(f"Could not find decisions for {conv_id}")
        
        problems = ''
        try:
            problist = xmldoc.getElementsByTagName('problems')
            items_sentences = problist[0].getElementsByTagName('sentence')
            for item in items_sentences:
                problems += item.firstChild.data + ' '
        except:
            print(f"Could not find problems for {conv_id}")
            
        return " ".join([summary,decisions,actions,problems]),summary, decisions, actions,problems

    def process_all(self,gen_transcript=True,gen_abssumm=True,gen_extsumm=True):
        conv_ids = [x.replace(".abssumm.xml","") for x in os.listdir(f"{self.ami_dir}/abstractive/")]
        abs_summaries = []
        abs_abstracts = []
        abs_decisions = []
        abs_actions = []
        abs_problems = []
        ext_summaries = []
        transcripts = []


        for conv_id in conv_ids:
            if gen_transcript:
                transcripts.append(f"{conv_id} {self.get_transcript(conv_id)}")
            if gen_abssumm:
                summary,abstract,decision,action,problems = self.get_abssum(conv_id)
                abs_summaries.append(f"{conv_id} {summary}")
                abs_abstracts.append(f"{conv_id} {abstract}")
                abs_decisions.append(f"{conv_id} {decision}")
                abs_actions.append(f"{conv_id} {action}")
                abs_problems.append(f"{conv_id} {problems}")
            
            # if gen_extsumm:
            #     ext_summaries.append(get_extsumm(data_root,conv_id))

        out_dir = os.path.join("data","processed_text")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(os.path.join(out_dir,"transcript"),"w") as f0, open(os.path.join(out_dir,"abstract"),"w") as f1, open(os.path.join(out_dir,"decision"),"w") as f2, open(os.path.join(out_dir,"action"),"w") as f3, open(os.path.join(out_dir,"problem"),"w") as f4, open(os.path.join(out_dir,"abs_summary"),"w") as f5:
            f0.write("\n".join(transcripts))
            f1.write("\n".join(abs_abstracts))
            f2.write("\n".join(abs_decisions))
            f3.write("\n".join(abs_actions))
            f4.write("\n".join(abs_problems))
            f5.write("\n".join(abs_summaries))




   

    # """ Begin 3/4: EXTRACTIVE SUMMARY     """
    # def extract_extractive_summary(self):
    #     """ Extract summary from each meeting
    #           * Located in `data/ami_public_manual_1.6.2/extractive/*.xml`
    #     """
    #     print("\nObtaining extractive summaries to: {}".format(self.args.results_summary_dir))
    #     sum_dir = self.ami_dir + '/extractive/'
    #     if not os.path.exists(self.args.results_summary_dir):
    #         os.mkdir(self.args.results_summary_dir)
    #     results_extractive_summary_dir = '{}/{}/'.format(self.args.results_summary_dir, "extract")
    #     sum_files = [f for f in os.listdir(sum_dir) if f.endswith('.extsumm.{}'.format(self.in_file_ext))]
    #     for sum_filename in sum_files:
    #         summary = self.extract_extractive_summary_single_file(sum_dir, sum_filename, results_extractive_summary_dir)
    #     print("Summaries: {}".format(len(sum_files)))

    # def extract_extractive_summary_single_file(self, summary_dir, summary_filename, results_dir):
    #     """ Obtain extractive summary (ALL sentences/highlights) from one meeting (one file)
    #           * Extract text between `extsumm` tag
    #           * Text between `extsumm` tag is composed of children nodes such as the below example:
    #             * `<nite:child href="ES2002a.B.dialog-act.xml#id(ES2002a.B.dialog-act.dharshi.3)"/>`
    #             * This node refers to a a node of ID `ES2002a.B.dialog-act.dharshi.3` in a file named `ES2002a.B.dialog-act.xml` in `data/dialogueActs/`
    #             * This ID is:
    #             ```
    #             <dact nite:id="ES2002a.B.dialog-act.dharshi.3" reflexivity="true">
    #                 <nite:pointer role="da-aspect"  href="da-types.xml#id(ami_da_4)"/>
    #                 <nite:child href="ES2002a.B.words.xml#id(ES2002a.B.words4)..id(ES2002a.B.words16)"/>
    #             </dact>
    #             ```
    #             * This indicates words 4 to 16 in `ES2002a.B.words.xml`
    #           * Return all the collected words as a paragraph

    #     :param summary_dir: directory containing .xml files with meeting abstract/summary
    #     :param summary_filename: name of each summary file
    #     :param results_dir: directory to save .txt files with summaries
    #     :return:
    #     """
    #     # parse an xml file by name
    #     mydoc = minidom.parse(summary_dir + summary_filename)
    #     mydoc = mydoc.childNodes[0].childNodes[1]  # getElementsByTagName('extsumm')
    #     items_extsumm = mydoc.childNodes
    #     summary = ""
    #     for items_ext in items_extsumm:
    #         if items_ext.attributes is not None:  # Element
    #             filename, init_id, final_id = self.obtain_ids(items_ext)
    #             summary += self.obtain_dialogue_act_node_references(filename, init_id, final_id)

    #     # Save summary
    #     # results_dir = self.args.results_summary_dir
    #     results_filename = summary_filename.replace('.{}'.format(self.in_file_ext), '.{}'.format(self.out_file_ext))
    #     self.save_file(summary, results_dir, results_filename)

    #     return summary

    # def obtain_dialogue_act_node_references(self, filename, init_id, final_id):
    #     """ Obtain node references from extractive summary file

    #     :param filename:
    #     :param init_id: initial ID for extractive summary
    #     :param final_id: final ID for extractive summary
    #     :return:
    #     """
    #     sum_dir = self.ami_dir + '/dialogueActs/'

    #     # Obtain array with dialogAct IDs
    #     print(init_id)
    #     id_arr = [init_id]
    #     if final_id is not None:
    #         init_id_index = int(init_id.split('dialog-act.')[-1].split('.')[-1])
    #         final_id_index = int(final_id.split('dialog-act.')[-1].split('.')[-1])
    #         root_name = final_id.split(str(final_id_index))[0]
    #         for i in range(init_id_index+1, final_id_index+1):
    #             id_arr.append(root_name + str(i))

    #     # parse an xml file by name
    #     segment_nodes = self.obtain_node_from_textname(sum_dir + filename, id_arr, tag='dact')

    #     words = self.obtain_dialogue_act_node(segment_nodes)
    #     return words

    

    # def obtain_ids(self, items_ext):
    #     href = items_ext.getAttribute('href')
    #     filename = href.split('#')[0]
    #     id_tmp = href.split('#')[-1].split(')..id(')
    #     init_id = id_tmp[0].split('id(')[-1].split(')')[0]
    #     final_id = None
    #     if len(id_tmp) > 1:
    #         final_id = id_tmp[1].split(')')[0]
    #     return filename, init_id, final_id

    # def obtain_node_from_textname(self, path, id_arr, tag):
    #     mydoc = minidom.parse(path)
    #     items_dact = mydoc.getElementsByTagName(tag)
    #     nodes = []
    #     for item in items_dact:
    #         nite_id = item.getAttribute('nite:id')
    #         if nite_id in id_arr:
    #             nodes.append(item)
    #     return nodes
    
    # """ End: EXTRACTIVE SUMMARY     """

    # """ Begin 4/4: FORMAT SUMMARY INTO .STORY     """
    # def check_for_meeting_summary(self, meeting_name, summary_files):
    #     """ Check if summary exists for a certain meeting

    #     :param meeting_name: prefix indicating meeting name
    #     :param summary_files: existing summary files
    #     :return:
    #     """
    #     for s in summary_files:
    #         if meeting_name in s:
    #             return s
    #     return None

    
    # def save_file(self, text, dir, filename):
    #     if not os.path.exists(dir):
    #         os.mkdir(dir)
    #     file = open(dir + filename, 'w')
    #     file.write(text)
    #     file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts transcript and summary from AMI Corpus.')
    parser.add_argument('--ami_dir', type=str, default='/ocean/projects/iri120008p/roshansh/corpora/ami/',
                        help='AMI Corpus download directory')
       
    args = parser.parse_args()

    # 1. Downloads original AMI Corpus and saves in `data/ami_public_manual_1.6.2`
    processor = AMIDataExtractor(args.ami_dir)

    # 2. Extract transcript for each subject (words/*.xml -> meeting transcripts)
    processor.process_all(gen_transcript=True,gen_abssumm=True, gen_extsumm=True)

    # # 3. Extract summary
    # if args.summary_type == "abstract":
    #     AMIDataExtractor.extract_abstractive_summary()
    # else:  # extractive
    #     AMIDataExtractor.extract_extractive_summary()