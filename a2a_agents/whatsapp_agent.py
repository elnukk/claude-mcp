import json

def create_whatsapp_message(alert_json: dict) -> dict:
    """
    Creates a rich formatted WhatsApp message from a structured alert JSON.
    """

    message = (
        f"🚨 *Weather Alert* 🚨\n\n"
        f"📍 *Location:* {alert_json['location']['village']}, {alert_json['location']['district']}\n"
        f"🌾 *Crop:* {alert_json['crop']['name'].capitalize()} ({alert_json['crop']['stage']})\n"
        f"⚠️ *Urgency:* {alert_json['alert']['urgency'].upper()}\n\n"
        f"📝 *Details:* {alert_json['alert']['message']}\n\n"
        f"✅ *Recommended Actions:*\n"
    )
    for action in alert_json['alert']['action_items']:
        message += f"- {action.replace('_', ' ').capitalize()}\n"

    return {
        "text": message,
        "buttons": [
            {"title": "Acknowledge", "payload": f"ack_{alert_json['alert_id']}"},
            {"title": "More Info", "payload": f"info_{alert_json['alert_id']}"}
        ]
    }

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
        "stage": "flowering",
        "planted_estimate": "2025-06-15"
      },
      "alert": {
        "type": "weather_warning",
        "urgency": "high",
        "message": "Heavy rainfall (40-60mm) expected in next 2 days. Delay fertilizer application. Ensure proper drainage.",
        "action_items": ["delay_fertilizer", "check_drainage"],
        "valid_until": "2025-07-25T18:00:00Z"
      },
      "weather": {
        "forecast_days": 3,
        "rain_probability": 85,
        "expected_rainfall": "45mm"
      }
    }

    whatsapp_message = create_whatsapp_message(sample_alert)
    print(json.dumps(whatsapp_message, indent=2))
