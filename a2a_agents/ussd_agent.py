def create_ussd_menu(alert_json: dict) -> str:
    """
    Creates a USSD menu from a structured alert JSON.
    """

    menu = (
        "Mausam ki jankari:\n"
        f"1. {alert_json['crop']['name'].capitalize()} ki chetavani\n"
        "2. Salah\n"
        "3. Exit"
    )
    return menu

def get_ussd_submenu(alert_json: dict, choice: int) -> str:
    """
    Returns a submenu based on the user's choice.
    """
    if choice == 1:
        return (
            f"Chetavani: {alert_json['alert']['message']}\n"
            "0. Back"
        )
    elif choice == 2:
        actions = "\n".join([f"- {action.replace('_', ' ').capitalize()}" for action in alert_json['alert']['action_items']])
        return (
            f"Salah:\n{actions}\n"
            "0. Back"
        )
    else:
        return "Invalid choice. Please try again."

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

    main_menu = create_ussd_menu(sample_alert)
    print("--- Main Menu ---")
    print(main_menu)

    submenu_1 = get_ussd_submenu(sample_alert, 1)
    print("\n--- Submenu 1 ---")
    print(submenu_1)

    submenu_2 = get_ussd_submenu(sample_alert, 2)
    print("\n--- Submenu 2 ---")
    print(submenu_2)
