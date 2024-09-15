import time
import json
import torch

class ExecutionTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.events = [{},{},{},{}]
        self.tags = []
        self.recorder = {}

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
            torch.cuda.synchronize()
            current_time = time.time()
            self.tags.append((tag,current_time))

    def pop(self):
        if self.start_time != None and self.end_time == None:
            if self.tags != []:
                popped_tag, start_time = self.tags.pop()
                torch.cuda.synchronize()
                now = time.time()
                past_time = now - start_time
                global_rank = torch.cuda.current_device()
                
                if self.events[global_rank].get(popped_tag) != None:
                    self.events[global_rank][popped_tag] += past_time
                    self.recorder[popped_tag]["counter"] += 1
                    self.recorder[popped_tag]["max_time"] = max(self.recorder[popped_tag]["max_time"], past_time)
                    self.recorder[popped_tag]["min_time"] = min(self.recorder[popped_tag]["min_time"], past_time)
                    self.recorder[popped_tag]["accumulate_time"] += past_time
                else:
                    self.events[global_rank][popped_tag] = past_time
                    self.recorder[popped_tag] = {"counter": 1, "max_time": past_time, "min_time": past_time, "accumulate_time": past_time}
            else:
                print("No tags to pop")

    def _save_to_file(self, filepath):
        data = {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'events': self.events,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    def get_tag_record(self, tag):
        return self.recorder[tag]

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
