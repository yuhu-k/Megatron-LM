# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests


if __name__ == "__main__":
    url = sys.argv[1]
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}

    while True:
        sentence = "How to use pytorch to write an easy model training script?\n"#input("Enter prompt: ")
        tokens_to_generate = 2048#int(eval(input("Enter number of tokens to generate: ")))

        data = {"prompts": [sentence], "tokens_to_generate": tokens_to_generate}
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
        else:
            print("Megatron Response: ")
            print(response.json()["text"][0])
        break
