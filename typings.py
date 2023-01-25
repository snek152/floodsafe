import typing

class HourlyData(typing.TypedDict):
    time: list[str]
    temperature_2m: list[float]

class APIResponse(typing.TypedDict):
    latitude: float
    longitude: float
    generationtime_ms: float
    utc_offset_seconds: int
    timezone: str
    elevation: float
    hourly: HourlyData

