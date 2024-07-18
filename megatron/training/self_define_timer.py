import time
import json
import torch

class ExecutionTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.events = [{},{},{},{}]
        self.tags = []
        self.tmp = {}
        self.rank = []

    def start(self):
        self.start_time = time.time()

    def end(self, filepath):
        self.end_time = time.time()
        print("Start saving the profiling result")
        self._save_to_file(filepath)
        if self.tags != []:
            print("Error, the tags stack isn't clear.", self.tags)

    def push(self, tag):
        if self.start_time != None and self.end_time == None:
            current_time = time.time()
            self.tags.append((tag,current_time))

    def pop(self):
        if self.start_time != None and self.end_time == None:
            if self.tags != []:
                popped_tag, start_time = self.tags.pop()
                now = time.time()
                past_time = now - start_time
                global_rank = torch.distributed.get_rank()
                
                if self.events[global_rank].get(popped_tag) != None:
                    self.events[global_rank][popped_tag] += past_time
                    self.tmp[popped_tag].append((start_time,now,past_time))
                else:
                    self.events[global_rank][popped_tag] = past_time
                    self.tmp[popped_tag] = [(start_time,now,past_time)]
            else:
                print("No tags to pop")

    def _save_to_file(self, filepath):
        data = {
            'rank': self.rank,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'events': self.events,
            'log': self.tmp
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

# Example usage
# timer = ExecutionTimer()
# timer.start()
# timer.push("event1")
# time.sleep(2)
# timer.pop()
# timer.push("event2")
# time.sleep(1)
# timer.pop()
# timer.end("execution_time.json")
