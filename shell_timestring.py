import datetime
import pytz


def generate_timestamp():
    est = pytz.timezone('America/New_York')
    utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    est_now = utc_now.astimezone(est)
    print(est_now.strftime("%Y%m%d_%H%M%S_%f"))


if __name__ == "__main__":
    generate_timestamp()
