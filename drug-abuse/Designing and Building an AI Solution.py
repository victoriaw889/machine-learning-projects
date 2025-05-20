from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Sample training data 

substance_abuse = ["I love fenty", "I have overdosed before", "I felt like I was melting into the universe.", "Time didn’t exist. I was stuck in a loop, but I liked it.", "The trees were breathing, and I swear they were watching me.", "I cleaned my whole house at 3am and then reorganized my files alphabetically.", "I couldn’t stop talking — my brain was going a mile a minute.", "I wrote 10 pages of my essay in two hours and didn’t blink once.", "No hunger. No sleep. Just productivity.", "I love drugs."]
no_substance_abuse = ["I don't joke with the possibility of addiction", "Bad experience.", "I am very sad.", "Don't fall into addiction.", "Drugs kill dreams.", "Drugs: not even once.", "I love minecraft.", "Are you on something?", "It was almost like I was on drugs."] 

text_data = substance_abuse + no_substance_abuse 
text_labels = [1] * len(substance_abuse) + [0] * len(no_substance_abuse)  # 1 = positive, 0 = negative 

# Step 1: Vectorize the training data 

vectorizer = CountVectorizer() 
text_training = vectorizer.fit_transform(text_data) 

# Step 2: Initialize and train the Naive Bayes classifier 

text_classifier = MultinomialNB() 
text_classifier.fit(text_training, text_labels) 

# Step 3: Transform the intercepted text and make predictions 

intercepted_text = "drugs" #<-----------------PUT YOUR INPUT HERE!!!!!!!!!!!!!
text_counts = vectorizer.transform([intercepted_text]) 

# Predict probabilities 

final_pos = text_classifier.predict_proba(text_counts)[0][1] 
final_neg = text_classifier.predict_proba(text_counts)[0][0] 

# Output the result 

if final_pos > final_neg: 
    print("Drug abuse detected.") 
else: 
    print("No drug abuse detected.") 
    
text_data = substance_abuse + no_substance_abuse
text_labels = [1] * len(substance_abuse) + [0] * len(no_substance_abuse)

X_train, X_test, y_train, y_test = train_test_split(text_data, text_labels, test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

text_classifier = MultinomialNB()
text_classifier.fit(X_train_counts, y_train)

y_pred = text_classifier.predict(X_test_counts)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Step 6: Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Abuse", "Substance Abuse"]))

# Step 7: Confusion Matrix with Labels
labels = ["No Abuse", "Substance Abuse"]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()