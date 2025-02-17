import src
from chatbot import CB
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/src")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return src.chat(userText)
    
app.run(debug = True)
