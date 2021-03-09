from pipelines import pipeline

nlp=pipeline("e2e-qg")
ques = nlp("Python is a programming language. Created by Zino and first released in 1991")
print(ques)