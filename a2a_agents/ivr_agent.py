def create_ivr_script(alert_json: dict) -> list[dict]:
    """
    Creates a voice script with timing from a structured alert JSON.
    """

    script = [
        {"text": f"Namaste. Mausam ki chetavani {alert_json['location']['district']} ke liye.", "delay_after": 1},
        {"text": f"Fasal: {alert_json['crop']['name']}.", "delay_after": 1},
        {"text": f"Chetavani: {alert_json['alert']['message']}", "delay_after": 2},
        {"text": "Salah ke liye, ek dabaye.", "delay_after": 0}
    ]
    return script

def get_ivr_submenu_script(alert_json: dict) -> list[dict]:
    """
    Returns a submenu script for the IVR.
    """
    actions = ". ".join([action.replace('_', ' ') for action in alert_json['alert']['action_items']])
    script = [
        {"text": f"Salah: {actions}", "delay_after": 2},
        {"text": "Dhanyavad.", "delay_after": 0}
    ]
    return script

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

    main_script = create_ivr_script(sample_alert)
    print("--- Main Script ---")
    for line in main_script:
        print(line)

    submenu_script = get_ivr_submenu_script(sample_alert)
    print("\n--- Submenu Script ---")
    for line in submenu_script:
        print(line)
