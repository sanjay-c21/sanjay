"""
Dataset Generator for Sentiment Analysis of Media Posts.
Generates a comprehensive dataset with diverse social media posts, reviews,
comments, and sentences across Positive, Negative, and Neutral classes.
"""

import os
import pandas as pd
import random

def generate_dataset(output_path="datasets/text_dataset.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    positive_samples = [
        "I really enjoyed this movie. It was amazing!",
        "This product is absolutely incredible, works like magic!",
        "Had a wonderful time with friends today, blessed! 😊",
        "Best purchase I have ever made this year, highly recommended.",
        "The customer service was outstanding and extremely helpful.",
        "Loved the cinematography and soundtrack of the film. A masterpiece!",
        "So happy with my results today! Hard work pays off 🚀",
        "Delicious food and exceptional ambiance, 10/10!",
        "Super fast shipping and top notch quality packaging.",
        "This app has transformed my daily productivity completely.",
        "Kudos to the entire development team for this wonderful update.",
        "Great experience from start to finish, will definitely come back!",
        "Feeling energetic and ready to conquer the week ahead ✨",
        "The battery life on this phone is truly unbelievable.",
        "Such an inspiring and heartwarming story, touched my soul.",
        "Brilliant performance by the lead actors, standing ovation!",
        "Clean, intuitive design and super responsive user interface.",
        "Everything was delivered in perfect condition ahead of schedule.",
        "I am so grateful for all the support and kindness today.",
        "Five stars! Exceeded all my expectations.",
        "The concert last night was pure electric euphoria 🎸",
        "Phenomenal sound quality with crystal clear treble and deep bass.",
        "Beautiful sunset view from my balcony this evening 🌅",
        "I cannot stop recommending this book to everyone I meet.",
        "Flawless execution and very professional handling.",
        "Such a joyful celebration with the whole family!",
        "The new features in this update are so smooth and handy.",
        "Fantastic atmosphere, friendly staff, and mouthwatering desserts.",
        "Got promoted today! Thrilled and excited for the new journey 🎉",
        "Very easy to install and works straight out of the box.",
        "Splendid craftsmanship and premium feel in the hand.",
        "One of the best decisions I have made, totally worth every penny.",
        "The camera captures breathtaking detail even in low light.",
        "Warmest greetings and lots of love to everyone having a good day.",
        "Impressive turnaround time and very courteous support agent.",
        "The plot twist at the end blew my mind! Absolutely genius.",
        "Super comfortable shoes, walked 10 miles without any pain.",
        "My heart is full of gratitude and happiness today ❤️",
        "What a thrilling win for our team in the championship finals!",
        "Exceptionally well-written code and spotless documentation.",
        "I adore how sleek and modern this looks on my desk.",
        "Generous discounts and great value for money.",
        "The graphics in this game are stunning and immersive.",
        "Smooth transactions and instant confirmation, perfect service.",
        "So proud of my sister for graduating with honors! 🎓",
        "The coffee here is brewed to absolute perfection.",
        "Brilliant innovation that solves a real everyday problem.",
        "Super helpful tutorial, explained complex topics with such ease.",
        "Feeling blessed and excited for what tomorrow holds.",
        "Truly a five star experience in hospitality and comfort."
    ]
    
    negative_samples = [
        "This product is terrible. Complete waste of money!",
        "Worst customer support ever, waited two hours and got disconnected.",
        "Very disappointed with the build quality, broke within two days.",
        "The movie was so boring and predictable. Huge letdown.",
        "Food was cold, stale, and smelled awful. Disgusting!",
        "The app keeps crashing constantly after the latest buggy update.",
        "Horrible experience, I will never buy from this seller again.",
        "Extremely slow delivery and package arrived badly crushed.",
        "I feel terrible and exhausted after that frustrating meeting.",
        "Total scam, the item looks nothing like the picture online.",
        "Misleading advertisements and hidden fees everywhere.",
        "Terrible battery drain, drops from 100% to 20% in two hours.",
        "The hotel room was dirty, noisy, and had no hot water.",
        "Rude and unhelpful staff who refused to assist with my return.",
        "Lost all my unsaved work because the software froze completely.",
        "Don't waste your time or hard earned money on this junk.",
        "Sad to see how downhill this restaurant has gone over time.",
        "The controls are clunky, laggy, and unresponsive.",
        "Awful sound quality with annoying buzzing and distortion.",
        "I regret buying this item, worst decision ever made.",
        "So frustrated with the constant server outages today 😡",
        "The story was depressing, incoherent, and pointless.",
        "Received the wrong size and seller is ignoring all my emails.",
        "Extremely painful headache and feeling sick all afternoon.",
        "High price for extremely cheap and fragile plastic.",
        "Cancelled my flight without prior notice or compensation.",
        "The wifi is completely down again for the third time this week.",
        "Terrible acting, cheesy dialogues, and horrible CGI.",
        "My account got locked for no reason and zero support available.",
        "Poorly engineered and dangerous to use around children.",
        "Never felt so disrespected by a company's representative.",
        "The taste was bitter, bland, and inedible. Threw it away.",
        "Buggy, laggy, and completely unoptimized software.",
        "Such a sad and heartbreaking situation unfolding today.",
        "Defective unit right out of the box, refuses to power on.",
        "Empty promises and deceptive marketing practices.",
        "The fabric feels rough, itchy, and shrank in the first wash.",
        "Terrible delay at the airport with zero information provided.",
        "Unhappy with the outcome after spending so many hours.",
        "The screen flickers uncontrollably and hurts my eyes.",
        "Loud and obnoxious neighbors kept me awake all night.",
        "Misplaced my luggage and nobody is taking responsibility.",
        "Completely useless feature that only adds unnecessary confusion.",
        "Disgusted by the poor hygiene standards at this venue.",
        "Horrific customer journey with confusing and broken links.",
        "So angry about this sudden unjust price increase.",
        "Boring lecture with an unengaging and monotone speaker.",
        "The tire went flat on the highway during heavy rain.",
        "Subscribed by mistake and they refuse to issue a refund.",
        "Deeply upset and let down by how things were handled."
    ]
    
    neutral_samples = [
        "The package arrived today at 3:00 PM.",
        "The meeting has been rescheduled for next Tuesday at 10 AM.",
        "The product contains 100% cotton and is machine washable.",
        "The train leaves platform 4 at 14:30.",
        "Please find attached the monthly financial report for review.",
        "The store is open from Monday to Saturday from 9 AM to 8 PM.",
        "The temperature in the city today is 24 degrees Celsius.",
        "The recipe calls for two cups of flour and one teaspoon of salt.",
        "Software version 2.4.1 was released earlier this morning.",
        "The flight duration from New York to London is roughly 7 hours.",
        "Users can reset their password by visiting the settings tab.",
        "The library operates on weekdays with restricted weekend hours.",
        "The device connects via Bluetooth 5.0 and USB Type-C.",
        "The conference will feature speakers from various tech companies.",
        "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
        "The bus stops at main street every fifteen minutes.",
        "This document outlines the standard operating procedures.",
        "The report was compiled using data from the previous quarter.",
        "The car has driven 45,000 miles since manufacture.",
        "There are 24 hours in a standard solar day.",
        "The new policy will take effect starting next month.",
        "The package weighs approximately 1.5 kilograms.",
        "Press the power button once to turn the device on or off.",
        "The museum is located near the central railway station.",
        "Today is Wednesday, the third week of September.",
        "The lecture covers fundamental concepts of linear algebra.",
        "The document is available in PDF, DOCX, and TXT formats.",
        "The battery is currently charged to 65 percent.",
        "The speed limit on this section of the highway is 60 mph.",
        "The article discusses historical events from the 19th century.",
        "The system has been updated with the latest security patches.",
        "Registration closes on Friday at midnight.",
        "The table dimensions are 120 cm length by 60 cm width.",
        "The video has a runtime of 45 minutes and 30 seconds.",
        "All visitors are required to sign in at the front reception.",
        "The survey consists of ten multiple-choice questions.",
        "The printer is connected to the local wireless network.",
        "The package includes a user manual and a warranty card.",
        "The elevator is undergoing scheduled maintenance today.",
        "The current exchange rate is 1 USD to 83.5 INR.",
        "Classes will resume following the national holiday break.",
        "The file size is 250 megabytes.",
        "The restaurant serves both vegetarian and non-vegetarian options.",
        "The parcel was deposited in your mailbox.",
        "The next train arrives in approximately 5 minutes.",
        "The book contains twelve chapters and an index.",
        "The server is operating within normal temperature parameters.",
        "The meeting room is booked for the marketing presentation.",
        "The experiment was conducted under controlled laboratory conditions.",
        "This is an automated notification regarding your recent order."
    ]

    # Expansion templates to generate a rich, realistic dataset of 2,000+ examples
    pos_modifiers = [
        "I have to say, {}", "Honestly, {}", "Without a doubt, {}", 
        "Just wanted to post that {}", "Super glad that {}", "Wow, {}",
        "So true: {}", "Review update: {}", "Can confirm that {}", "Amazing: {}"
    ]
    neg_modifiers = [
        "I am so annoyed that {}", "Avoid this: {}", "Warning: {}", 
        "Sadly, {}", "I cannot believe {}", "Fed up: {}", 
        "Frustrated: {}", "Terrible: {}", "Review warning: {}", "Disappointed: {}"
    ]
    neu_modifiers = [
        "Note that {}", "Notice: {}", "Update: {}", "FYI, {}", 
        "Official statement: {}", "Informational: {}", "Fact: {}", 
        "System message: {}", "Status log: {}", "Schedule update: {}"
    ]
    
    data = []
    
    # Base samples
    for text in positive_samples:
        data.append({"text": text, "sentiment": "Positive"})
    for text in negative_samples:
        data.append({"text": text, "sentiment": "Negative"})
    for text in neutral_samples:
        data.append({"text": text, "sentiment": "Neutral"})
        
    # Augmented variations
    for _ in range(12):
        for text in positive_samples:
            mod = random.choice(pos_modifiers)
            data.append({"text": mod.format(text.lower()), "sentiment": "Positive"})
        for text in negative_samples:
            mod = random.choice(neg_modifiers)
            data.append({"text": mod.format(text.lower()), "sentiment": "Negative"})
        for text in neutral_samples:
            mod = random.choice(neu_modifiers)
            data.append({"text": mod.format(text.lower()), "sentiment": "Neutral"})
            
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} samples saved to {output_path}")
    print(df["sentiment"].value_counts())
    return df

if __name__ == "__main__":
    generate_dataset()
