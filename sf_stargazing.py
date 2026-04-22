"""Naked-eye stargazing condition scorer for San Francisco using Tomorrow.io + ephem."""

import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import ephem

API_KEY = "t4I7QP1xYuLZBhjIbYPiVQPF8udeEER9"
HOURLY_URL = f"https://api.tomorrow.io/v4/timelines?apikey={API_KEY}"
HOURLY_PAYLOAD = {
    "location": "37.77,-122.42",
    "fields": ["cloudCover", "humidity", "windSpeed", "visibility", "precipitationIntensity"],
    "timesteps": ["1h"],
    "timezone": "America/Los_Angeles",
    "startTime": "nowPlus0h",
    "endTime": "nowPlus48h",
}
THRESHOLD = 80
LA = ZoneInfo("America/Los_Angeles")

SF = ephem.Observer()
SF.lat = "37.77"
SF.lon = "-122.42"
SF.elevation = 50


def fetch(url: str, payload: dict) -> dict:
    body = json.dumps(payload)
    result = subprocess.run(
        [
            "curl", "-s", "--compressed", "--max-time", "10",
            "-X", "POST", url,
            "-H", "content-type: application/json",
            "-H", "accept: application/json",
            "-d", body,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode \!= 0:
        raise RuntimeError(f"curl failed: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    if "code" in data and data.get("code") \!= 200:
        raise RuntimeError(f"API error {data.get('code')}: {data.get('message', data)}")
    return data


def parse_intervals(data: dict) -> list[dict]:
    intervals = data["data"]["timelines"][0]["intervals"]
    result = []
    for iv in intervals:
        t = iv["startTime"]
        date_str = t[:10]
        hour_str = t[11:13]
        v = iv["values"]
        result.append({
            "date": date_str,
            "hour": hour_str,
            "cloudCover": v.get("cloudCover", 0),
            "humidity": v.get("humidity", 0),
            "windSpeed": v.get("windSpeed", 0),
            "visibility": v.get("visibility", 24.14),
            "precipitationIntensity": v.get("precipitationIntensity", 0),
        })
    return result


def get_window_hours(intervals: list[dict], date_str: str) -> dict[str, list]:
    keys = ["cloudCover", "humidity", "windSpeed", "visibility", "precipitationIntensity"]
    result: dict[str, list] = {k: [] for k in keys}
    for iv in intervals:
        if iv["date"] == date_str and iv["hour"] in ("21", "22", "23"):
            for k in keys:
                result[k].append(iv[k])
    return result


def get_precip_hours(intervals: list[dict], date_str: str) -> list[float]:
    return [
        iv["precipitationIntensity"]
        for iv in intervals
        if iv["date"] == date_str and iv["hour"] in ("15", "16", "17", "18", "19", "20")
    ]


def moon_data_for_date(local_date: date) -> tuple[float, float]:
    moon = ephem.Moon()
    midpoint_local = datetime(local_date.year, local_date.month, local_date.day, 22, 0, tzinfo=LA)
    midpoint_utc = midpoint_local.astimezone(ZoneInfo("UTC"))
    moon.compute(midpoint_utc.strftime("%Y/%m/%d %H:%M:%S"), epoch="2000")
    illumination = moon.moon_phase * 100

    obs = SF.copy()
    noon_local = datetime(local_date.year, local_date.month, local_date.day, 12, 0, tzinfo=LA)
    obs.date = noon_local.astimezone(ZoneInfo("UTC")).strftime("%Y/%m/%d %H:%M:%S")

    try:
        rise_utc = obs.next_rising(ephem.Moon()).datetime()
        rise_local = rise_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(LA)
        rise_min = rise_local.hour * 60 + rise_local.minute
    except ephem.NeverUpError:
        return illumination, 0.0
    except ephem.AlwaysUpError:
        rise_min = 0

    try:
        set_utc = obs.next_setting(ephem.Moon()).datetime()
        set_local = set_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(LA)
        set_min = set_local.hour * 60 + set_local.minute
        if set_min < rise_min:
            set_min = 1440
    except ephem.NeverUpError:
        set_min = rise_min
    except ephem.AlwaysUpError:
        set_min = 1440

    overlap = max(0, min(set_min, 1440) - max(rise_min, 1260))
    return illumination, overlap / 180


def score_night(
    window: dict[str, list],
    precip_pre: list[float],
    illumination: float,
    moon_frac: float,
) -> tuple[int, list[str]]:
    if not window["cloudCover"]:
        raise ValueError("no window data")

    avg_cloud = sum(window["cloudCover"]) / len(window["cloudCover"])
    avg_humidity = sum(window["humidity"]) / len(window["humidity"])
    avg_wind_ms = sum(window["windSpeed"]) / len(window["windSpeed"])
    avg_vis_km = sum(window["visibility"]) / len(window["visibility"])

    cloud_score = 50 * max(0, (1 - avg_cloud / 100) ** 1.5)
    vis_score = 15 * min(avg_vis_km, 24.14) / 24.14
    humidity_score = 15 * max(0, (1 - avg_humidity / 100) ** 0.8)
    wind_score = 10 * max(0, 1 - avg_wind_ms / 10)
    moon_penalty = (illumination / 100) * moon_frac * 20

    precip_sum = sum(precip_pre)
    post_rain_bonus = 10 if (precip_sum > 0.5 and avg_cloud < 40) else 0

    raw = cloud_score + vis_score + humidity_score + wind_score - moon_penalty + post_rain_bonus
    score = max(0, min(100, round(raw)))

    factors: list[tuple[float, str]] = [
        (cloud_score / 50, "clear skies"),
        ((vis_score + humidity_score) / 30, "excellent transparency"),
        (wind_score / 10, "calm winds"),
    ]
    factors.sort(reverse=True)
    top_labels = [label for _, label in factors[:2]]
    if post_rain_bonus and "post-rain clarity" not in top_labels:
        top_labels = top_labels[:1] + ["post-rain clarity"]

    return score, top_labels


def moon_summary(illumination: float, moon_frac: float) -> str:
    illum_int = round(illumination)
    if illum_int < 10:
        return "new moon"
    if moon_frac == 0:
        return "moon down all night"
    return f"moon {illum_int}% lit, up {moon_frac * 100:.0f}% of window"


def format_message(prefix: str, score: int, factors: list[str], moon_str: str) -> str:
    parts = factors + [moon_str]
    return f"{prefix}: {score}/100 — {', '.join(parts)}"


def main() -> None:
    try:
        hourly_data = fetch(HOURLY_URL, HOURLY_PAYLOAD)
        intervals = parse_intervals(hourly_data)

        now = datetime.now(LA)
        today = now.date()
        tomorrow = today + timedelta(days=1)
        today_str = today.strftime("%Y-%m-%d")
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")

        tonight_window = get_window_hours(intervals, today_str)
        tonight_precip = get_precip_hours(intervals, today_str)
        t_illum, t_frac = moon_data_for_date(today)
        tonight_score, tonight_factors = score_night(tonight_window, tonight_precip, t_illum, t_frac)
        tonight_moon = moon_summary(t_illum, t_frac)

        tomorrow_window = get_window_hours(intervals, tomorrow_str)
        tomorrow_precip = get_precip_hours(intervals, tomorrow_str)
        tm_illum, tm_frac = moon_data_for_date(tomorrow)
        tomorrow_score, tomorrow_factors = score_night(tomorrow_window, tomorrow_precip, tm_illum, tm_frac)
        tomorrow_moon = moon_summary(tm_illum, tm_frac)

        if tonight_score >= THRESHOLD:
            msg = format_message("Stargazing tonight", tonight_score, tonight_factors, tonight_moon)
            if tomorrow_score >= THRESHOLD:
                msg += f" (tomorrow also looks good: {tomorrow_score}/100)"
            print(msg)
        elif tomorrow_score >= THRESHOLD:
            print(format_message("Stargazing tomorrow night", tomorrow_score, tomorrow_factors, tomorrow_moon))

    except Exception as exc:
        print(f"Stargazing check failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
