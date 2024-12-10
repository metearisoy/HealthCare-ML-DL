import openai

openai.api_key = "api_key"

def chat_with_gpt(prompt, history):
    
   responce = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role":"user",
                     "content": f"You are a doctor, specifically a family physician. Our past conversations: {history}, new question: {prompt}"}]
   )
   
   return responce.choices[0].message.content.strip()

if __name__ == "__main__":
    history = []
    
    while True:
        
        user_input = input(" what is your message?")
        
        if user_input.lower() in ["exit", ""]:
            print("The correspondence is completed.")
            break
        
        history.append(user_input)
        responce = chat_with_gpt(user_input, history)
        print("Chatbot: ",responce )
    
    