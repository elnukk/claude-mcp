import asyncio
from src.mcp_weather_server.tools import geographic_tools

class LocationAgent:
    async def get_locations(self, state, district):
        print(f"Fetching villages for {district}, {state}...")
        villages_response = await geographic_tools.list_villages(state=state, district=district)
        if "error" in villages_response:
            print(f"Error: {villages_response['error']}")
            return None

        locations = []
        for village in villages_response["villages"]:
            print(f"Getting coordinates for {village}...")
            coordinates_response = await geographic_tools.reverse_geocode(location_name=village)
            if "error" in coordinates_response:
                print(f"Error: {coordinates_response['error']}")
                continue

            locations.append({
                "village": village,
                "latitude": coordinates_response["latitude"],
                "longitude": coordinates_response["longitude"]
            })
        return locations

async def main():
    agent = LocationAgent()
    locations = await agent.get_locations(state="bihar", district="patna")
    if locations:
        print("\n--- Locations ---")
        for loc in locations:
            print(f"Village: {loc['village']}, Lat: {loc['latitude']}, Lon: {loc['longitude']}")
        print("-----------------")

if __name__ == "__main__":
    asyncio.run(main())
