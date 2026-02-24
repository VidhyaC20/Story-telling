print("App is starting...")
import os
import warnings
from flask import Flask, render_template, request, send_file
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from flask_cors import CORS


warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=3000
)

prompt = ChatPromptTemplate.from_template("""
You are a professional storyteller and narrator.

Generate a detailed and complete narration about the topic.
Make it natural, clear, and suitable for audio.
Only deliver the content strictly.
The story must be complete.

Topic:
{topic}
""")

chain = prompt | llm


def generate_text(topic: str) -> str:
    response = chain.invoke({"topic": topic})
    return response.content


def text_to_audio(text: str, output_file="static/output_story.mp3"):
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        input=text
    ) as response:
        response.stream_to_file(output_file)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        topic = request.form["topic"]

        generated_text = generate_text(topic)
        text_to_audio(generated_text)

        return render_template("index.html",
                               story=generated_text,
                               audio_file="static/output_story.mp3")

    return render_template("index.html")


if __name__ == "__main__":

    app.run(debug=True)
