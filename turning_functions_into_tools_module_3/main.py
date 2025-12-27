from datetime import datetime
from langchain_core.tools import tool
from llm.llm_client_using_langchain_with_tool import LLMClientTool

import requests
import qrcode
from qrcode.image.styledpil import StyledPilImage



@tool
def get_current_time_tool() -> str:
    """Returns the current time in HH:MM:SS format"""
    return datetime.now().strftime("%H:%M:%S")


@tool
def get_sum_tool(a: int, b: int) -> int:
    """Returns the sum of a and b"""
    return a+b



@tool
def get_weather_from_ip():
    """
    Gets the current, high, and low temperature in Fahrenheit for the user's
    location and returns it to the user.
    """
    # Get location coordinates from the IP address
    lat, lon = requests.get('https://ipinfo.io/json').json()['loc'].split(',')

    # Set parameters for the weather API call
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone": "auto"
    }

    # Get weather data
    weather_data = requests.get("https://api.open-meteo.com/v1/forecast", params=params).json()

    # Format and return the simplified string
    return (
        f"Current: {weather_data['current']['temperature_2m']}°F, "
        f"High: {weather_data['daily']['temperature_2m_max'][0]}°F, "
        f"Low: {weather_data['daily']['temperature_2m_min'][0]}°F"
    )



@tool
# Write a text file
def write_txt_file(file_path: str, content: str):
    """
    Write a string into a .txt file (overwrites if exists).
    Args:
        file_path (str): Destination path.
        content (str): Text to write.
    Returns:
        str: Path to the written file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path



import os
import qrcode
from qrcode.image.styledpil import StyledPilImage
from langchain_core.tools import tool


@tool
def generate_qr_code(
    data: str,
    logo_path: str | None = None
) -> str:
    """
    Generates a QR code.
    If logo_path is provided, embeds the logo.
    """

    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)

    if logo_path:
        if not os.path.isfile(logo_path):
            img = qr.make_image(fill_color="black", back_color="white")
        else:
            img = qr.make_image(
                image_factory=StyledPilImage,
                embedded_image_path=os.path.abspath(logo_path)
            )

    else:
        img = qr.make_image(fill_color="black", back_color="white")

    output_path = "qr_output.png"
    img.save(output_path)

    return f"QR code created at {output_path}"






LLMClientTool.add_tool(get_current_time_tool)
LLMClientTool.add_tool(get_sum_tool)
LLMClientTool.add_tool(generate_qr_code)
LLMClientTool.add_tool(write_txt_file)
LLMClientTool.add_tool(get_weather_from_ip)



# print(LLMClientTool.ask("what time is it?"))
# print(LLMClientTool.ask("what is sum of 1000 and 30000"))
# print(LLMClientTool.ask("Can you get the weather for my location?"))
print(LLMClientTool.robust_ask("Can you help me create a qr code that goes to www.deeplearning.com from the image dl_logo.jpg? Also write me a txt note with the current weather please."))



# from IPython.display import Image, display
# # Display image directly
# Image('dl_qr_code.png')