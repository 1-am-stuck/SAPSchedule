
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
    print("From:", message.sender)
    print("Subject:", message.subject)
   # print("Date:", message.date)
    #print("Body:", message.plain)  # plain text body
    if c>20:
        break
    print("-" * 50)

print(body[1:2])
print(subject[1:2])