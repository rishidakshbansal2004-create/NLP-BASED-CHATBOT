import pickle
import json
import random
from preprocessing import preprocess

with open("intent.json", "r") as f:
    data = json.load(f)

def get_response(user_input,model, vectorizer):
    clean_input= " ".join(preprocess(user_input))
    x=vectorizer.transform([clean_input])
    intent=model.predict(x)[0]
    scores = model.decision_function(x)
    conf=scores.max()
    if conf<0.3:
        intent="fallback"
    for i in data["intents"]:
        if i["tag"]==intent:
            return random.choice(i["responses"])
    
    return "SORRY! TRY AGAIN PLEASE"


#ONLY FOR TESTING  
if __name__ == "__main__":
    import pickle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "q": #q to exit
            break
        print(f"Bot: {get_response(user_input, model, vectorizer)}")

