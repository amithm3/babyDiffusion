import json
import os

if __name__ == '__main__':
    with open("captions.json", "r") as f:
        captions = json.load(f)

    files = os.listdir()
    files.remove("captions.json")
    files.remove("captions.py")

    pre_prompt = [
        """
        I will provide you with a list of files you must generate a list of captions inferred from the file name.
        i will give the input in the following format: [file1, file2, file3, ...]
        respond in a json format: [caption1, caption2, caption3, ...]
        """,
        """
        Dont always start and end in the same style.
        Create variations amongst the captions, do not use the same pattern for all captions.
        Use some level of creativity, but dont deviate too much.
        Do not include the numeric dimension in the caption.
        Dont always start with "A".
        """
    ]

    print(pre_prompt[0])
    print("---x---")
    print(pre_prompt[1])
    print("---x---")
    un_caped = [file for file in files if file not in captions]
    while len(un_caped) > 0:
        files = [un_caped.pop() for _ in range(min(25, len(un_caped)))]
        print(files)
        caped = eval(input("caped: "))
        for file in files:
            captions[file] = caped.pop(0)

    with open("captions.json", "w") as f:
        json.dump(captions, f, indent=4)
