import datetime

def get_date():
    """returns the date and time as string in the following form: yy-mm-dd-hh-mm"""
    return datetime.datetime.now().strftime("%y-%m-%d-%H-%M")