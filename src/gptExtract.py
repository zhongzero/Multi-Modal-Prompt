import openai
from openai import OpenAI


def process_sentence(sentence):
    pos_Foreground = sentence.find("Foreground: ")
    pos_Background = sentence.find("\nBackground: ")
    pos_Action = sentence.find("\nAction: ")
    pos_EndAction = len(sentence)
    foreground = sentence[pos_Foreground+12:pos_Background]
    background = sentence[pos_Background+13:pos_Action]
    action = sentence[pos_Action+9:pos_EndAction]
    if foreground[0] == '"':
        foreground = foreground[1:-1]
    if background[0] == '"':
        background = background[1:-1]
    if action[0] == '"':
        action = action[1:-1]
    return foreground, background, action


def gpt_txtExtract(raw_sentence, args):
    print(f"raw_sentence: {raw_sentence}")

    input = (
        f"I will give you some examples. "
        f"Given a sentence \"arafed dog sitting on the beach with its tongue out\", the foreground is \"arafed dog with its tongue out\", the background is \"beach\" and the action is \"sitting\". "
        f"Given a sentence \"there is a cat and a dog that are playing together\", the foreground is \"a cat and a dog\", the background is \"None\" and the action is \"playing\". "
        f"Given a sentence \"there are two cats that are playing with each other\", the foreground is \"two cats\", the background is \"None\" and the action is \"playing with each other\". "
        f"Given a sentence \"there is a cat on the bench\", the foreground is \"a cat\", the background is \"bench\" and the action is \"None\". "
        f"Given a sentence \"there is a white sandy beach with cat\", the foreground is \"a white sandy beach with cat\", the background is \"None\" and the action is \"None\". "
        f"Now imitate these, I will give you a sentence \"{raw_sentence}\", and you need to give me the foreground, background and action."
    )
    # input = (
    #     f"I will give you some examples.\n"
    #     f"Given a sentence \"arafed dog sitting on the beach with its tongue out\", the foreground is \"arafed dog with its tongue out\", the background is \"beach\" and the action is \"sitting\".\n"
    #     f"Given a sentence \"there is a cat and a dog that are playing together\", the foreground is \"a cat and a dog\", the background is \"None\" and the action is \"playing\".\n"
    #     f"Given a sentence \"there are two cats that are playing with each other\", the foreground is \"two cats\", the background is \"None\" and the action is \"playing with each other\".\n"
    #     f"Given a sentence \"there is a cat on the bench\", the foreground is \"a cat\", the background is \"bench\" and the action is \"None\".\n"
    #     f"Given a sentence \"there is a white sandy beach with cat\", the foreground is \"a white sandy beach with cat\", the background is \"None\" and the action is \"None\".\n"
    #     f"Now imitate these, I will give you a sentence \"{raw_sentence}\", and you need to give me the foreground, background and action. The format is \"<Foreground: foreground> <Background: background> <Action: action>\"."
    # )
    # input = (
    #     f"Objective: Analyze a given sentence to identify the foreground, background, and action elements.\n"
    #     f"Examples: \n"
    #     f"Sentence: \"A ragged dog sitting on the beach with its tongue out\" output: <Foreground: \"a ragged dog with its tongue out\" Background: \"beach\" Action: \"sitting\">\n"
    #     f"Sentence: \"There is a cat and a dog that are playing together\" output: <Foreground: \"a cat and a dog\" Background: None Action: \"playing\">\n"
    #     f"Sentence: \"There are two cats that are playing with each other\" output: <Foreground: \"two cats\" Background: None Action: \"playing with each other\">\n"
    #     f"Sentence: \"There is a cat on the bench\" output: <Foreground: \"a cat\" Background: \"bench\" Action: None>\n"
    #     f"Sentence: \"There is a white sandy beach with a cat\" output: <Foreground: \"a white sandy beach with a cat\" Background: None Action: None>\n"
    #     f"imitate examples above:\n"
    #     f"Sentence: \"{raw_sentence}\", output: \n"
    # )
    print(input)
    messages = []
    messages.append({"role": "user", "content": input})
    max_try_times = 10
    try_times = 0
    str2numMap = {}
    lasans = None
    while try_times < max_try_times:
        try:
            client = OpenAI(
                api_key=args.gpt_api_key,
                base_url=args.gpt_base_url,
                timeout=30
            )
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages)
        except Exception as e:
            if isinstance(e, openai.APITimeoutError):
                print("Failed to connect to gpt-3.5-turbo")
                try_times += 1
                continue
            else:
                raise e
        answer = completion.choices[0].message.content
        print(answer)
        
        pos_Foreground = answer.find("Foreground: ")
        pos_Background = answer.find("\nBackground: ")
        pos_Action = answer.find("\nAction: ")
        if pos_Foreground != -1 and pos_Background != -1 and pos_Action != -1 and pos_Foreground < pos_Background and pos_Background < pos_Action:
            if lasans == answer:
                return process_sentence(answer)
            else:
                lasans = answer
                if answer in str2numMap:
                    str2numMap[answer] += 1
                else:
                    str2numMap[answer] = 1
        try_times += 1
    if str2numMap == {}:
        raise Exception(
            "Failed to extract foreground, background and action from gpt-3.5-turbo")
    else:
        max_num = 0
        max_str = None
        for key in str2numMap:
            if str2numMap[key] > max_num:
                max_num = str2numMap[key]
                max_str = key
        return process_sentence(max_str)