import json

user_question = input("ask a question about us!!\n")

with open("data.json", "r") as f:
    faq_data = json.load(f)

for faq in faq_data:
    if user_question.lower() in faq['question'].lower():
        print(faq['answer'])