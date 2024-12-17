from langchain_community.document_loaders import CSVLoader

with open("data.csv", "r") as f:
    faq_data = f.read()

print(faq_data)