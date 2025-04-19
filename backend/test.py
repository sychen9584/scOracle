from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="What's the population of Boston?"
)

print(response.output_text)