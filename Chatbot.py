import pickle
import json
import random
from preprocessing import preprocess
with open("model.pkl","rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)
with open("intent.json", "r") as f:
    data = json.load(f)

def get_response(user_input):
    clean_input= " ".join(preprocess(user_input))
    x=vectorizer.transform([clean_input])
    intent=model.predict(x)[0]
    conf=max(model.predict_proba(x)[0])
    if conf<0.3:
        intent="fallback"
    for i in data["intents"]:
        if i["tag"]==intent:
            return random.choice(i["responses"])
    
    return "SORRY! TRY AGAIN PLEASE"


#FOR TESTING 
if __name__ == "__main__":
    print("Chatbot is running! Type 'q' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "q":
            break
        response = get_response(user_input)
        print(f"Bot: {response}")

