import json

def to_hindi(text):
    """
    Mock function to "translate" text to Hindi.
    In a real implementation, this would use a translation service.
    """
    translations = {
        "Heavy rainfall": "भारी वर्षा",
        "expected in next 2 days.": "अगले 2 दिनों में अपेक्षित।",
        "Delay fertilizer application.": "उर्वरक आवेदन में देरी करें।",
        "Ensure proper drainage.": "उचित जल निकासी सुनिश्चित करें।",
        "rice": "चावल",
        "flowering": "फूल",
        "weather_warning": "मौसम की चेतावनी",
        "high": "उच्च",
        "Kumhrar": "कुम्हरार",
        "Patna": "पटना",
        "Bihar": "बिहार",
        "Alert": "चेतावनी",
        "Crop": "फसल",
        "Stage": "चरण",
        "Urgency": "तात्कालिकता",
        "Action": "कार्य"
    }
    for en, hi in translations.items():
        text = text.replace(en, hi)
    return text

def create_sms_message(alert_json: dict) -> str:
    """
    Creates a 160-character SMS message from a structured alert JSON.
    """

    message = (
        f"{to_hindi('Alert')}: {to_hindi(alert_json['alert']['type'])}, "
        f"{to_hindi('Crop')}: {to_hindi(alert_json['crop']['name'])}, "
        f"{to_hindi('Stage')}: {to_hindi(alert_json['crop']['stage'])}, "
        f"{to_hindi('Urgency')}: {to_hindi(alert_json['alert']['urgency'])}. "
        f"{to_hindi(alert_json['alert']['message'])}"
    )

    # Truncate to 160 characters
    return message[:160]

if __name__ == '__main__':
    sample_alert = {
      "alert_id": "BH_PAT_001_20250723",
      "timestamp": "2025-07-23T06:00:00Z",
      "location": {
        "village": "Kumhrar",
        "district": "Patna",
        "state": "Bihar",
        "coordinates": [25.5941, 85.1376]
      },
      "crop": {
        "name": "rice",
        "stage": "flowering"
      },
      "alert": {
        "type": "weather_warning",
        "urgency": "high",
        "message": "Heavy rainfall expected in next 2 days. Delay fertilizer application. Ensure proper drainage."
      }
    }

    sms = create_sms_message(sample_alert)
    print(sms)
    print(f"Length: {len(sms)}")
