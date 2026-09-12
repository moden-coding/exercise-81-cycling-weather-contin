#!/usr/bin/env python3

import contextlib
import io
import re
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import sklearn

from src.cycling_weather_continues import cycling_weather_continues, main


def spy_decorator(method_to_decorate, name):
    """
    Wrap a method so calls to it are recorded on a MagicMock while the
    original implementation still runs.

    This solution to wrap a patched method without obstructing its
    implementation comes originally from
    https://stackoverflow.com/questions/25608107/
    """
    mock = MagicMock(name="%s method" % name)

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)
    wrapper.mock = mock
    return wrapper


class TestCyclingWeatherContinues(unittest.TestCase):

    def test_return_type(self):
        self.coef, self.score = cycling_weather_continues("Merikannontie")
        self.assertIsInstance(
            self.coef, (list, tuple, np.ndarray),
            msg="Expected coefficients to be a list, tuple or an array!")
        self.assertIsInstance(
            self.score, float, msg="Expected the score to be a float!")

        self.assertAlmostEqual(
            self.score, 0.66, places=2,
            msg="Incorrect score for station 'Merikannontie'!")
        self.assertAlmostEqual(
            self.coef[0], -58.2, places=1,
            msg="Incorrect regression coefficient for precipitation!")
        self.assertAlmostEqual(
            self.coef[1], -15.8, places=1,
            msg="Incorrect regression coefficient for precipitation!")
        self.assertAlmostEqual(
            self.coef[2], 145.6, places=1,
            msg="Incorrect regression coefficient for precipitation!")

    def test_output(self):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            main()
        output = buf.getvalue().strip()
        self.assertRegex(
            output, r"(?m)Measuring station: *(.+)$",
            msg="No information about the measuring station in output!")
        self.assertRegex(
            output,
            r"(?m)Regression coefficient for variable 'precipitation': "
            r"[-+]?\d+\.\d$",
            msg="Incorrect output for variable precipitation!")
        self.assertRegex(
            output,
            r"(?m)Regression coefficient for variable 'snow depth': "
            r"[-+]?\d+\.\d$",
            msg="Incorrect output for variable snowdepth!")
        self.assertRegex(
            output,
            r"(?m)Regression coefficient for variable 'temperature': "
            r"[-+]?\d+\.\d$",
            msg="Incorrect output temperature!")
        self.assertRegex(
            output, r"(?m)Score: [-+]?\d+\.\d\d$",
            msg="Incorrect output about score!")

    def test_calls(self):
        merge_method = spy_decorator(pd.core.frame.DataFrame.merge, "merge")
        with patch("src.cycling_weather_continues.cycling_weather_continues",
                   wraps=cycling_weather_continues) as pcw, \
             patch("src.cycling_weather_continues.LinearRegression",
                   wraps=sklearn.linear_model.LinearRegression) as lr, \
             patch("src.cycling_weather_continues.pd.read_csv",
                   wraps=pd.read_csv) as prc, \
             patch.object(pd.core.frame.DataFrame, "merge", new=merge_method), \
             patch("src.cycling_weather_continues.pd.merge",
                   wraps=pd.merge) as pmerge:
            main()
            pcw.assert_called_once()
            lr.assert_called_once()
            if "fit_intercept" in lr.call_args[1]:
                self.assertTrue(
                    lr.call_args[1]["fit_intercept"],
                    msg="You did not fit the intercept!")
            elif len(lr.call_args[0]) > 0:
                self.assertTrue(
                    lr.call_args[0][0],
                    msg="You did not fit the intercept!")
            merge_method_called = merge_method.mock.call_count >= 1
            merge_func_called = pmerge.call_count >= 1
            self.assertTrue(
                merge_method_called or merge_func_called,
                msg="You did not call merge method or function!")
            self.assertEqual(
                prc.call_count, 2,
                msg="You should have called pd.read_csv exactly twice")


if __name__ == '__main__':
    unittest.main()
