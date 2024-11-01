from flask import Flask, render_template, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer as SIA
sia = SIA()

app = Flask(__name__)

# Show main page
@app.route('/')
def index():

    return render_template('index.html')


# Process data which was posted through js
@app.route('/process', methods=['POST'])
def process():
    # Store the user input
    data = request.form['user_input']

    # Process the input
    data=Positive_Maker(data)

    # Sentiment Analysis
    scores = sia.polarity_scores(data)


    # Send the modified text + its scores back to the js script.
    return jsonify({'output':data,
        'neg':scores['neg'],
        'pos':scores['pos'],
        'neu':scores['neu']})


from nltk.sentiment import SentimentIntensityAnalyzer as SIA
import textwrap
from nltk.tokenize import word_tokenize, sent_tokenize
import re


dictionary = {'disgusting': 'fantastic', 'awful': 'excellent', 'hate': 'love', 'hateful': 'wonderful', 'weird': 'normal', 'arrogant': 'humble',
             'undesirable': 'desirable', 'heartless': 'compassionate', 'insane': 'rational', 'messy': 'organized','bad': 'good', 'worst': 'best','ignorant': 'intelligent',
              'narcissistic': 'unselfish', 'rude': 'polite', 'gross':'great','brutal': 'gentle', 'cheap': 'admirable','stupid': 'clever', 'ugly': 'beautiful',
              'unvaluable': 'valuable', 'boring': 'interesting','cruel': 'compassionate', 'dishonest': 'honest', 'harsh':'pleasant', 'jealous': 'trustful',
              'narrow-minded': 'open-minded', 'untrustworthy': 'trustworthy', 'unhappy': 'happy','impatient': 'patient','selfish': 'generous', 'egoistic': 'modest',
              'silly':'thoughtful', 'moody': 'stable','cowerd': 'fearless', 'impolite': 'polite', 'aloof': 'boon', 'bigoted': 'broad-minded','fussy':'patient',
              'grumpy': 'good-tempered', 'violent': 'calm', 'antisocial':'social', 'naive':'experienced', 'obnoxious':'acceptable', 'possessive': 'permissive',
             'sadistic':'compassionate','manipulative': 'honest', 'hostile': 'friendly'}



# List = ["not", "do", "n't", "does", "n't",]

def change_words(txt, dictionary):
    return " ".join([dictionary.get(w, w) for w in word_tokenize(txt)])

def Positive_Maker(data):
    txt = data
    sia = SIA()
    debug = True # Set this to True if you want to see intermediate steps of the text processing
    scores = sia.polarity_scores(txt)


    ### Step 1: Remove negations ###

    if scores['neg'] > 0.3:

        if debug: print('Input in step 1:', txt)

        # Optional: Store input in a text file on your disk
        save_negative_sentence(txt)

        # Change words like don't and doesn't before the tokenization happens
        txt = txt.replace('not', '')
        txt = txt.replace("don't", '')
        txt = txt.replace("n't", '')

        txt= txt.replace('  ', ' ') # Remove double spaces

        scores = sia.polarity_scores(txt)

        if debug: print('Output of step 1:', txt, '\n')

        ### Step 2: Change words ###

        # Check if step 1 is enough
        # Otherwise change words
        if scores['neg'] > 0.3:

            if debug: print('Input in step 2:', txt)

            txt = change_words(txt, dictionary)
            txt= re.sub(r'\s([?.!,"](?:\s|$))', r'\1', txt)
            scores = sia.polarity_scores(txt)

            if debug: print('Output of step 2:', txt, '\n')


            ### Step 3: Remove whole sentences if necessary ###

            # Check if step 2 is enough,
            # otherwise analyze individual sentences and remove bad ones entirely.
            if scores['neg'] > 0.3:

                if debug: print('Input in step 3:', txt)

                # Split txt into sentences
                sentences = sent_tokenize(txt)
                cleaned_txt = [] # A list to store the positive sentences
                for sentence in sentences:
                    scores = sia.polarity_scores(sentence)
                    # Append sentence to cleaned_txt if the score is below a threshold
                    if scores['neg'] < 0.3:
                        cleaned_txt.append(sentence)


                # Join cleaned_txt
                txt = ' '.join(cleaned_txt)

                if debug: print('Output of step 3:', txt, '\n')

    # Return modified text
    return txt

# This function appends a sentence to a (daily) text file
# It can be used in operation to save the user inputs for
# further improvement etc.
def save_negative_sentence(sentence):
    from pathlib import Path
    from datetime import datetime
    date = datetime.now().strftime('%Y_%m_%d') # like 2022_08_23

    # Path for writing the file
    source_path = Path(__file__).resolve()
    source_dir = str(source_path.parent)
    # Append sentence to a file (creates the file if it does not exist)
    with open(source_dir + '/negative_inputs_' + date + '.txt', 'a') as f:
        f.write(sentence)
        f.write('\n')


if __name__ == '__main__':
    app.debug = True # Set this to False when the development is done.
    app.run(port=7000)
