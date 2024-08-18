import datetime
import pytz


def get_timestring(time_string="%Y%m%d_%H%M%S_%f"):
    est = pytz.timezone('America/New_York')
    utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    est_now = utc_now.astimezone(est)
    return est_now.strftime(time_string)
