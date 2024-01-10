"""Regular expressions, part 1."""

import re  # python's builtin regular expression module

exercises = {
    1: (
        "TODO_replace_me_with_a_regex",
        [
            ("Match", "3.14529"),
            ("Match", "-255.34"),
            ("Match", "128"),
            ("Match", "1.9e10"),
            ("Match", "123,340.00"),
            ("Skip", "720p"),
        ],
    ),
    2: (
        "TODO_replace_me_with_a_regex",
        [
            ("Capture", "415-555-1234", ("415",)),
            # TODO add other tests...
        ],
    ),
}


def check_expression(reg_expr, test_type, test_string, capture_groups=None) -> bool:
    """Test regexone.com problem solutions.

    Parameters
    ----------
    reg_expr : str
        The regular expression to test
    test_type : str
        Match, Skip, or Capture
    test_str : str
        String to test regular expression on
    capture_groups : tuple
        Groups that should be captured by the regular expression.
    """
    pass  # TODO write this function


if __name__ == "__main__":
    for i, (reg_expr, tests) in exercises.items():
        for test in tests:
            assert check_expression(reg_expr, *test), f"Exercise {i}: {test} failed"
