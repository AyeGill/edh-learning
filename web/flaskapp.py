from flask import Flask, request
from flask import render_template

import sys
sys.path.insert(0, '/home/eigil/projects/edh-learning/')

from datasetutils import *
from deck_predictor_indep import *
app = Flask(__name__)

def strip_cmdr(card):
    if card[-1:]=='C':
        return card[:-2]
    else:
        return card
dic,rev = read_vocab('../vocabulary.txt')
cprobs = np.load('../cprobs.npy',mmap_mode='r')
probs = np.load('../probs.npy',mmap_mode='r')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/submit', methods=['POST'])
def submit():
    lines = [line.strip() for line in request.form.get('text').split('\n')]
    ids = [dic[line] for line in lines]
    outdeck = [rev[card] for card in predict_deck_f(ids,probs,cprobs)]
    oldcards = [(strip_cmdr(card),card) for card in lines]
    cards = [(strip_cmdr(card),card) for card in outdeck]
    return render_template('deck-render.html',cards=cards,oldcards=oldcards)