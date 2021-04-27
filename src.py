import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import gtts
from io import BytesIO
import speech_recognition as sr
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from playsound import playsound

with open("intents.json") as file:
	data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs_x.append(wrds)
		docs_y.append(intent["tag"])

	if intent["tag"] not in labels:
		labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)


training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
	bag = []

	wrds = [stemmer.stem(w) for w in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)
	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	training.append(bag)
	output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)


tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	model.load("model.tflearn")
except:
	model = tflearn.DNN(net)
	model.fit(training, output, n_epoch=500, batch_size=8, show_metric=True)
	model.save("model.tflearn")

def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]


	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

def chat(request):

	inp = request

	results = model.predict([bag_of_words(inp,words)])[0]
	results_index = numpy.argmax(results)
	tag = labels[results_index]

	if results[results_index] > 0.5:
		for tg in data["intents"]:
			if tg['tag'] == tag:
				responses = tg['responses']

		# tts = gtts.gTTS(random.choice(responses), tld="co.uk")
		# tts.save("temp.mp3")

		# playsound("temp.mp3")

		return (random.choice(responses))
	

	else:
		return ("I didnt get that, try again")


# r = sr.Recognizer()
# mic = sr.Microphone(device_index=0)

def initialize_chat():
	print("Start Talking with the bot")
	while True:
		print("Ask me anything: ")
		with mic as source:
			r.adjust_for_ambient_noise(source)
			audio = r.listen(source)
		result = r.recognize_google(audio)
		if result == "quit" or result == "exit" or result == "stop":
			print("Thanks for using the bot, exiting....")
			break
		chat(result)

# initialize_chat()