from unittest import TestCase
import datetime
import date_funcs as df

class TestDateFuncs(TestCase):
    """Test for the module date_funcs.py"""

    def test_get_date(self):
        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        self.assertEquals(date[:-2], df.get_date()[:-2])
