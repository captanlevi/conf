import datetime
from zoneinfo import ZoneInfo

def convertUNIXToHumanReadable(timestamp : float):
    return datetime.datetime.fromtimestamp(timestamp,tz=ZoneInfo("Australia/Sydney")).isoformat()