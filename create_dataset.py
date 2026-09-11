import pandas as pd
import random

positive = [
    "I love this product",
    "This product is amazing",
    "The service is excellent",
    "I am very happy with the purchase",
    "The experience was wonderful",
    "This application is very useful",
    "The product quality is fantastic",
    "I really enjoyed the service",
    "The application works perfectly",
    "I am satisfied with the product",
    "This is an excellent experience",
    "The service was great",
    "I would recommend this product",
    "Everything is working perfectly",
    "I am extremely happy",
    "The application is impressive",
    "The product exceeded my expectations",
    "The service was fast and excellent",
    "I really like this application",
    "This is a very good product"
]

negative = [
    "I hate this product",
    "This product is terrible",
    "The service is horrible",
    "I am very unhappy with the purchase",
    "The experience was disappointing",
    "This application is useless",
    "The product quality is awful",
    "I did not enjoy the service",
    "The application does not work",
    "I am disappointed with the product",
    "This is a terrible experience",
    "The service was very bad",
    "I would not recommend this product",
    "Everything went wrong",
    "I am extremely disappointed",
    "The application is frustrating",
    "The product failed completely",
    "The service was slow and terrible",
    "I really dislike this application",
    "This is a very bad product"
]

neutral = [
    "The product is available online",
    "The service is available today",
    "I purchased this product",
    "The application has several features",
    "The product has a standard design",
    "I received the package",
    "The service starts at nine",
    "The application was updated today",
    "I am using the application",
    "The product arrived today",
    "The service operates during the day",
    "The application contains several options",
    "I opened the application",
    "The product is listed on the website",
    "The service is currently available",
    "I placed an order",
    "The package arrived yesterday",
    "The application requires an account",
    "The product has different sizes",
    "The service is provided online"
]


def create_variations(sentences):
    variations = []

    prefixes = [
        "",
        "I think ",
        "In my opinion, ",
        "Overall, ",
        "Personally, ",
    ]

    suffixes = [
        "",
        ".",
        " today.",
        " for me.",
        " right now."
    ]

    for sentence in sentences:
        base = sentence.rstrip(".")
        variations.append(base)

        for prefix in prefixes[1:]:
            for suffix in suffixes[:3]:
                variations.append(prefix + base.lower() + suffix)

    return list(set(variations))


positive_data = create_variations(positive)
negative_data = create_variations(negative)
neutral_data = create_variations(neutral)

random.seed(42)

positive_data = random.sample(positive_data, min(100, len(positive_data)))
negative_data = random.sample(negative_data, min(100, len(negative_data)))
neutral_data = random.sample(neutral_data, min(100, len(neutral_data)))

data = []

for text in positive_data:
    data.append([text, "positive"])

for text in negative_data:
    data.append([text, "negative"])

for text in neutral_data:
    data.append([text, "neutral"])

random.shuffle(data)

df = pd.DataFrame(data, columns=["text", "sentiment"])

df.to_csv("data/sentiment_data.csv", index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print()
print(df["sentiment"].value_counts())