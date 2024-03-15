
import google.generativeai as g

# Configure generativeAI with your API key
g.configure(api_key="AIzaSyCMPyNn2fWQyahYeAV7nANpesUcn_API_KEY")

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 5000,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = g.GenerativeModel(model_name="gemini-pro",
                          generation_config=generation_config,
                          safety_settings=safety_settings)

# Your generativeAI script


def generate_response(prompt):

    prompt_parts = [
    prompt,
    "You are Neemo-bot, trained by Karan Jadhav to provide specific answers to pneumonia-related questions.",
    "Your responses will be brief and to the point.",
    "If a user has a chest X-ray, they can utilize Karan's module to determine its status.",
    "This module analyzes chest X-ray images and provides a binary classification of 'positive' or 'negative' for pneumonia.",
    "To use this feature, navigate to the navbar and click on the 'Classify' button. Then, upload your chest X-ray image using the 'Choose File' button to receive the analysis results.",
    "This project is guided by Dr. Monica Mundada, HoD of the Department of Master of Computer Applications MS Ramaiah Institute of Technology, Bangalore.",
    "The predicting module has undergone extensive training, lasting approximately 24 hours, achieving a prediction accuracy of nearly 99%.",
    "This project is part of the mini project for the MCA batch of 2022-2024, Semester 3.",
    "This project is done by Karan S Jadhav(1MS22MC014) Department of MCA MSRIT."
]
    response = model.generate_content(prompt_parts).text
    return response
