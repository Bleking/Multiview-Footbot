from google import genai

def summarize_video(question,video_path):
    client = genai.Client()
    interaction = client.interactions.create(
        model='gemini-3.8-flash',
        input=[
            {"type": "text", "text": question},
            {"type": "video", "uri": video_path}
        ]
    )
    
    return interaction.output_text

def generate_answer(prompt):
    client = genai.Client()
    interaction = client.interactions.create(
        model='gemini-3.8-flash',
        input=[
            {"type": "text", "text": prompt}
        ]
    )

    return interaction.output_text
