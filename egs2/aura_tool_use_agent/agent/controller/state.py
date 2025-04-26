from agent.dst.dst_action import DSTAction
class State:
    def __init__(self):
        self.history = []
        self.dst = None
        self.dst_class = None
        self.conversation = []

       
    def get_state(self):
        pass

    def get_tts_output(self):

        if self.dst_class is None:
            initial_dst_classify = DSTAction(thought='Have to get DST category', payload=self.history).execute(type='get_category')

            if initial_dst_classify['output'] is  None:
                self.dst_class = None
                #self.history.append({"role": "assistant", "content": initial_dst_classify['response']})
                return initial_dst_classify['response']
            else:
                self.dst_class = initial_dst_classify['output']
                print("***DST Class set to :", self.dst_class,'***')

            
        """We will have a valid DST categrory when we reach here
        We will now run a class specifc DST tracker from this point onwards so that the LLM finds it easier to keep track of the DST"""
        self.dst =  DSTAction(thought='Have to get DST for conversation', payload=self.history).execute(type=self.dst_class)
        return self.dst






                
                
           
        if self.dst_class is None:
            initial_dst_classify = DSTAction(thought='Have to get DST category', payload=self.history).execute(type='get_category')

            if initial_dst_classify['output'] is  None:
                self.dst_class = None
                #self.history.append({"role": "assistant", "content": initial_dst_classify['response']})
                return initial_dst_classify['response']
            else:
                self.dst_class = initial_dst_classify['output']
                print("***DST Class set to :", self.dst_class,'***')

            
        """We will have a valid DST categrory when we reach here
        We will now run a class specifc DST tracker from this point onwards so that the LLM finds it easier to keep track of the DST"""
        self.dst =  DSTAction(thought='Have to get DST for conversation', payload=self.history).execute(type=self.dst_class)
        return self.dst






                
                
           

        
