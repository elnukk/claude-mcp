"""
This module contains tools for crop calendar operations.
"""
from datetime import date

CROP_CALENDAR = {
    "rice": {
        "season": "Kharif",
        "planting": "June-July",
        "harvesting": "October-November",
        "stages": [
            "Nursery/Seedling", "Transplanting", "Vegetative", "Tillering",
            "Panicle Initiation", "Flowering", "Milk/Dough", "Maturity", "Harvesting"
        ]
    },
    "wheat": {
        "season": "Rabi",
        "planting": "November-December",
        "harvesting": "March-April",
        "stages": [
            "Sowing", "Germination", "Tillering", "Jointing", "Booting",
            "Heading", "Flowering", "Grain Filling", "Maturity", "Harvesting"
        ]
    },
    "maize": {
        "season": "Kharif/Zaid",
        "planting": "June-July / March-April",
        "harvesting": "September-October / June",
        "stages": [
            "Sowing", "Emergence", "Vegetative", "Tasseling", "Silking",
            "Grain Filling", "Maturity", "Harvesting"
        ]
    },
    "barley": {
        "season": "Rabi",
        "planting": "November",
        "harvesting": "March-April",
        "stages": [
            "Sowing", "Germination", "Tillering", "Jointing", "Booting",
            "Heading", "Flowering", "Grain Filling", "Maturity", "Harvesting"
        ]
    },
    "gram": {
        "season": "Rabi",
        "planting": "October-November",
        "harvesting": "March-April",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Pod Formation",
            "Pod Filling", "Maturity", "Harvesting"
        ]
    },
    "lentil": {
        "season": "Rabi",
        "planting": "October-November",
        "harvesting": "March-April",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Pod Formation",
            "Pod Filling", "Maturity", "Harvesting"
        ]
    },
    "pea": {
        "season": "Rabi",
        "planting": "October-November",
        "harvesting": "February-March",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Pod Formation",
            "Pod Filling", "Maturity", "Harvesting"
        ]
    },
    "mustard": {
        "season": "Rabi",
        "planting": "October-November",
        "harvesting": "February-March",
        "stages": [
            "Sowing", "Germination", "Rosette", "Stem Elongation", "Flowering",
            "Pod Formation", "Pod Filling", "Maturity", "Harvesting"
        ]
    },
    "linseed": {
        "season": "Rabi",
        "planting": "October-November",
        "harvesting": "March-April",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Capsule Formation",
            "Seed Filling", "Maturity", "Harvesting"
        ]
    },
    "potato": {
        "season": "Rabi",
        "planting": "October-November",
        "harvesting": "February-March",
        "stages": [
            "Planting", "Sprouting", "Vegetative", "Tuber Initiation", "Tuber Bulking",
            "Maturity", "Harvesting"
        ]
    },
    "arhar": {
        "season": "Kharif",
        "planting": "June-July",
        "harvesting": "November-December",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Pod Formation",
            "Pod Filling", "Maturity", "Harvesting"
        ]
    },
    "moong": {
        "season": "Kharif/Zaid",
        "planting": "June-July / March-April",
        "harvesting": "September-October / June",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Pod Formation",
            "Pod Filling", "Maturity", "Harvesting"
        ]
    },
    "urd": {
        "season": "Kharif/Zaid",
        "planting": "June-July / March-April",
        "harvesting": "September-October / June",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Pod Formation",
            "Pod Filling", "Maturity", "Harvesting"
        ]
    },
    "jowar": {
        "season": "Kharif",
        "planting": "June-July",
        "harvesting": "September-October",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Booting", "Flowering",
            "Grain Filling", "Maturity", "Harvesting"
        ]
    },
    "bajra": {
        "season": "Kharif",
        "planting": "June-July",
        "harvesting": "September-October",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Booting", "Flowering",
            "Grain Filling", "Maturity", "Harvesting"
        ]
    },
    "groundnut": {
        "season": "Kharif",
        "planting": "June-July",
        "harvesting": "September-October",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Pegging",
            "Pod Formation", "Pod Filling", "Maturity", "Harvesting"
        ]
    },
    "soybean": {
        "season": "Kharif",
        "planting": "June-July",
        "harvesting": "September-October",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Pod Formation",
            "Pod Filling", "Maturity", "Harvesting"
        ]
    },
    "watermelon": {
        "season": "Zaid",
        "planting": "March-April",
        "harvesting": "May-June",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Fruit Setting",
            "Fruit Development", "Maturity", "Harvesting"
        ]
    },
    "cucumber": {
        "season": "Zaid",
        "planting": "March-April",
        "harvesting": "May-June",
        "stages": [
            "Sowing", "Germination", "Vegetative", "Flowering", "Fruit Setting",
            "Fruit Development", "Maturity", "Harvesting"
        ]
    }
}


from datetime import date
from .geographic_tools import BIHAR_DATA

async def get_crop_calendar(region: str, crop_type: str = None):
    if region.lower() == 'bihar':
        if crop_type:
            crop = crop_type.lower()
            if crop in CROP_CALENDAR:
                return {**CROP_CALENDAR[crop], "crop": crop}
            else:
                return {"error": "Crop not found in Bihar"}
        return {
            "crops": list(CROP_CALENDAR.keys()),
            "districts": list(BIHAR_DATA.keys())
        }
    return {"error": "Region not found"}

async def get_prominent_crops(region: str, season: str):
    if region.lower() == 'bihar':
        season = season.lower()
        crops = [crop for crop, data in CROP_CALENDAR.items() if season in data["season"].lower()]
        if crops:
            return {"crops": crops}
        else:
            return {"error": "No crops found for this season in Bihar"}
    return {"error": "Region or season not found"}

async def estimate_crop_stage(crop: str, plant_date: str, current_date: str):
    plant_date = date.fromisoformat(plant_date)
    current_date = date.fromisoformat(current_date)
    days_since_planting = (current_date - plant_date).days

    crop = crop.lower()
    if crop in CROP_CALENDAR:
        stages = CROP_CALENDAR[crop]["stages"]
        # Estimate total duration (in days) for each crop (approximate, can be refined)
        crop_durations = {
            "rice": 120, "wheat": 120, "maize": 110, "barley": 110, "gram": 110,
            "lentil": 110, "pea": 100, "mustard": 110, "linseed": 110, "potato": 100,
            "arhar": 150, "moong": 70, "urd": 70, "jowar": 100, "bajra": 90,
            "groundnut": 110, "soybean": 100, "watermelon": 80, "cucumber": 70
        }
        total_duration = crop_durations.get(crop, 100)
        stage_length = total_duration // len(stages)
        stage_index = min(days_since_planting // stage_length, len(stages) - 1)
        return {"stage": stages[stage_index]}
    return {"error": "Crop not found"}