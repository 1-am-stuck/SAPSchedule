
from simplegmail import Gmail
from simplegmail.query import construct_query
from datetime import datetime

# Authenticate (credentials.json must be in the same folder)
gmail = Gmail(r'C:\Users\srija\OneDrive\Desktop\Project\gmail\client_secret.json')

# Example: Get all unread messages
messages = gmail.get_unread_inbox()
body = []
subject = []
c = 0
for message in messages:
    c = c+1
    body.append(message.plain)
    subject.append(message.subject)
   
   # print("Date:", message.date)
    #print("Body:", message.plain)  # plain text body
    if c>15:
        break



from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load from local folder
model_path = "./saved_model/bart-large-mnli"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
labels = ["Event", "Not Event"]
event=[]
def check(text):
    result = classifier(text, candidate_labels=labels)
    
    if result['scores'][0]>result['scores'][1]:
        return True

import re
import spacy
from dateutil import parser
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

date =[]
time = []


def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    #text = text.translate(str.maketrans('', '', string.punctuation))


    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # Join back into cleaned string
    cleaned_text = " ".join(tokens)
    text = cleaned_text



# Load English model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

def extract_event_details(text):
    details = {
        "event_name": None,
        "date": None,
        "time": None,
        "venue": None
    }

    # Process text with spaCy
    doc = nlp(text)

    # --- Find Event Name ---
    # Assume event name appears near words like "event", "conference", "meeting", etc.
    event_keywords = ["event", "conference", "meeting", "workshop", "seminar", "party", "festival", "ceremony"]
    for sent in doc.sents:
        if any(word.lower() in sent.text.lower() for word in event_keywords):
            details["event_name"] = sent.text.strip()
            break

    # --- Extract DATE & TIME ---
    dates = []
    times = []

    # Regex for times
    time_pattern = r'(\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?)|(\d{1,2}\s?(?:AM|PM|am|pm))'
    times_found = re.findall(time_pattern, text)
    for t in times_found:
        t = "".join(t).strip()
        if t:
            times.append(t)

    # Use spaCy NER for DATE entities
    for ent in doc.ents:
        if ent.label_ == "DATE":
            try:
                dt = parser.parse(ent.text, fuzzy=True, dayfirst=True)
                dates.append(str(dt.date()))
            except:
                pass

    if dates:
        details["date"] = dates[0]
    if times:
        details["time"] = times[0]

    # --- Extract VENUE ---
    # Look for GPE (Geo-Political Entities) or FAC (Facilities)
    venues = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "FAC", "ORG", "LOC"]]
    if venues:
        details["venue"] = venues[-1]  # Last location mentioned is often the venue
    date.append(str(details["date"]))
    print(date,'date')
    time.append(details["time"])
    events.append([subject[body.index(text)],date,time])

    #return details
events= []
for i in body:
    if check(i)==True:
        clean_text(i)
        extract_event_details(i)
print(events)
'''
for i in events:
    print(i)
    print('-'*50)
'''
'''print(date)
print(time)'''



     
