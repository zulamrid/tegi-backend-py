#import lib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Conversation, ConversationalPipeline
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
nlp = ConversationalPipeline(model=model, tokenizer=tokenizer)

conversation = Conversation()

#init flask
app = Flask(__name__)

#init cors
CORS(app)

@app.route('/')
def main():
    return "This is the main"

@app.route('/add_input', methods = ['GET', 'POST'])
def add_input():
     text = request.json['text']
     conversation.add_user_input(text)
     result = nlp([conversation], do_sample=False, max_length=1000)
     messages = []
     for is_user, text in result.iter_texts():
          messages.append({
               'is_user': is_user,
               'text': text
          })
     return jsonify({
          'uuid': result.uuid,
          'messages': messages
     })

@app.route('/reset', methods = ['GET', 'POST'])
def reset():
     global conversation
     conversation = Conversation()
     return 'ok' 

if __name__ == "__main__":
    app.run(debug=True, port=5000)