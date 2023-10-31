import pandas as pd
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
def get_vader_sentiment(text, analyzer):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']
def clean_and_tag(text):
    cleaned_text = re.sub(r'\W', ' ', str(text))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text, flags=re.I).lower()
    codes = [keyword_to_construct[word] for word in cleaned_text.split() if word in keyword_to_construct]
    for question, code in question_to_code.items():
        cleaned_question = re.sub(r'\W', ' ', str(question))
        cleaned_question = re.sub(r'\s+', ' ', cleaned_question, flags=re.I).lower()
        if cleaned_question in cleaned_text:
            codes.append(code)
    codes = list(set(codes))  # removing duplicates
    return pd.Series([cleaned_text, list(set(codes))])
#Given dictionaries
keyword_to_construct = {
    'enhancement': 'UTPE', 'productivity': 'UTPE', 'effectiveness': 'UTPE',
    'improvement': 'UTPE', 'advancement': 'UTPE', 'efficiency': 'UTPE',
    'innovation': 'UTPE', 'optimization': 'UTPE', 'benefit': 'UTPE',
    'ease': 'UTPE', 'convenience': 'UTPE', 'simplification': 'UTPE',
    'ease of use': 'UTEE',  'simplicity': 'UTEE', 'convenience': 'UTEE',
    'user-friendly': 'UTEE','effortless': 'UTEE', 'accessible': 'UTEE',
    'intuitive': 'UTEE', 'easy': 'UTEE', 'simple': 'UTEE', 'clear': 'UTEE',
     'compliance': 'UTSI',    'conformity': 'UTSI',    'social networks': 'UTSI',
    'impartiality disclosure': 'UTSI',    'influencers': 'UTSI',    'consumer responses': 'UTSI',
    'social media': 'UTSI', 'friends': 'UTSI', 'colleagues': 'UTSI', 'social': 'UTSI', 'group': 'UTSI',
     'adoption': 'UTFC',     'implementation': 'UTFC',    'utilization': 'UTFC',    'behavioral': 'UTFC',
    'moderators': 'UTFC',    'mediators': 'UTFC',    'voluntariness': 'UTFC', 'ease of use': 'UTFC',
    'sustainable': 'UTFC',    'internal agents': 'UTFC',    'external agents': 'UTFC',
    'support': 'UTFC', 'help': 'UTFC', 'facilitate': 'UTFC', 'assist': 'UTFC'}
question_to_code = {
    'Tell me about your thoughts on how a Metaverse system might influence your academic performance.': 'UTPE1',
    'Can you explain how you think a Metaverse system might aid in accomplishing tasks more swiftly?': 'UTPE2',
    'Describe how using a Metaverse system could affect your study efficiency.': 'UTPE3',
    'Share your feelings about learning and using a Metaverse system. Do you find it would be easy or challenging?': 'UTEE1',
    'Explain your thoughts on becoming proficient in using a Metaverse system.': 'UTEE2',
    'Discuss your confidence level in handling the technical aspects of a Metaverse system.': 'UTEE3',
    'Tell me about the people who influence your behavior and their opinion on you using a Metaverse system.': 'UTSI1',
    'How do you think using a Metaverse system would affect your social acceptance among peers?': 'UTSI2',
    'Share your thoughts on friends or family’s perspective on your use of a Metaverse system.': 'UTSI3',
    'Explain your thoughts on having the necessary resources and support to use a Metaverse system.': 'UTFC1',
    'Can you discuss the availability of resources and support for utilizing a Metaverse system?': 'UTFC2',
    'Share your thoughts on obtaining help or support while using a Metaverse system.': 'UTFC3'}
# Load the Dataset
Metaverse_df = pd.read_csv('Metaverse_Dataset.csv')
Metaverse_df[['cleaned_text', 'codes']] = Metaverse_df['text'].apply(clean_and_tag)
analyzer = SentimentIntensityAnalyzer()
Metaverse_df['sentiment_score'] = Metaverse_df['cleaned_text'].apply(lambda text: get_vader_sentiment(text, analyzer))
# Calculate average sentiments
avg_sentiments = {}
all_codes = set(code for sublist in Metaverse_df['codes'].tolist() for code in sublist)
for code in all_codes:
    sentiment_scores = Metaverse_df[Metaverse_df['codes'].apply(lambda codes: code in codes)]['sentiment_score']
    avg_sentiment = sentiment_scores.mean()
    if pd.isna(avg_sentiment):
        print(f"No entries for code {code}")
        avg_sentiments[code] = 0
    else:
        print(f"Average sentiment for code {code}: {avg_sentiment:.2f}")
        avg_sentiments[code] = avg_sentiment
# Plotting
plt.figure(figsize=(6, 3))
plt.bar(avg_sentiments.keys(), avg_sentiments.values(), color=['blue', 'green', 'yellow', 'orange'])
plt.xlabel('Codes')
plt.ylabel('Average Sentiment')
plt.title('Average Sentiment for Each Code')
plt.xticks(rotation=45)
plt.show()