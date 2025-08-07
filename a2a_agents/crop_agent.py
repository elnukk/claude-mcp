import asyncio
from src.mcp_weather_server.tools import crop_calendar_tools
import datetime

class CropAgent:
    def get_current_season(self):
        # This is a simplified way to determine the season.
        # A more robust implementation would use a more accurate method.
        month = datetime.datetime.now().month
        if month >= 10 or month <= 3:
            return "Rabi"
        else:
            return "Kharif"

    async def get_crop_info(self, state, season, plant_date, current_date):
        print(f"Finding prominent crops for {season} season in {state}...")
        prominent_crops_response = await crop_calendar_tools.get_prominent_crops(region=state, season=season)
        if "error" in prominent_crops_response:
            print(f"Error: {prominent_crops_response['error']}")
            return None

        crop_info_list = []
        for crop in prominent_crops_response["crops"]:
            print(f"Estimating crop stage for {crop} planted on {plant_date}...")
            crop_stage_response = await crop_calendar_tools.estimate_crop_stage(crop=crop, plant_date=plant_date, current_date=current_date)
            if "error" in crop_stage_response:
                print(f"Error: {crop_stage_response['error']}")
                continue

            crop_info_list.append({
                "crop": crop,
                "growth_stage": crop_stage_response["stage"]
            })
        return crop_info_list

async def main():
    agent = CropAgent()
    season = agent.get_current_season()
    print(f"Current season: {season}")

    # Example usage
    plant_date = "2023-11-01"
    current_date = "2024-02-15"
    crop_info = await agent.get_crop_info(state="bihar", season=season, plant_date=plant_date, current_date=current_date)

    if crop_info:
        print("\n--- Crop Info ---")
        for info in crop_info:
            print(f"Crop: {info['crop']}, Growth Stage: {info['growth_stage']}")
        print("-----------------")

if __name__ == "__main__":
    asyncio.run(main())
