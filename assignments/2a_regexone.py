"""Regular expressions, part 1."""

import re  # python's builtin regular expression module

exercises = {  # "Official" solutions in comments
    1: (
        r"^[-0-9e.,]+$",
        # r"^-?\d+(,\d+)*(\.\d+(e\d+)?)?$"
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
        r"^(?:1[-\s]?)?\(?(\d{3})\)?[-\s]?\d{3}[-\s]?\d{4}$",
        #      1?[\s-]?\(?(\d{3})\)?[\s-]?\d{3}[\s-]?\d{4}.
        [
            ("Capture", "415-555-1234", ("415",)),
            ("Capture", "650-555-2345", ("650",)),
            ("Capture", "(416)555-3456", ("416",)),
            ("Capture", "202 555 4567", ("202",)),
            ("Capture", "4035555678", ("403",)),
            ("Capture", "1 416 555 9292", ("416",)),
        ],
    ),
    3: (
        r"^([^+@]+)(?:\+[^@]+)?@[\w.]+$",
        # ^([\w\.]*)
        [
            ("Capture", "tom@hogwarts.com", ("tom",)),
            ("Capture", "tom.riddle@hogwarts.com", ("tom.riddle",)),
            ("Capture", "tom.riddle+regexone@hogwarts.com", ("tom.riddle",)),
            ("Capture", "tom@hogwarts.eu.com", ("tom",)),
            ("Capture", "potter@hogwarts.com", ("potter",)),
            ("Capture", "harry@hogwarts.com", ("harry",)),
            ("Capture", "hermione+regexone@hogwarts.com", ("hermione",)),
        ],
    ),
    4: (
        r"<(\w+)[^>]*>.*?</\1>",
        # <(\w+)
        [
            ("Capture", "<a>This is a link</a>", ("a",)),
            ("Capture", "<a href='https://regexone.com'>Link</a>", ("a",)),
            ("Capture", "<div class='test_style'>Test</div>", ("div",)),
            ("Capture", "<div>Hello <span>world</span></div>", ("div",)),
        ],
    ),
    5: (
        r"^([^.]*)\.(gif|jpg|png)$",
        #    (\w+)\.(jpg|png|gif)$
        [
            ("Skip", ".bash_profile"),
            ("Skip", "workspace.doc"),
            (
                "Capture",
                "img0912.jpg",
                (
                    "img0912",
                    "jpg",
                ),
            ),
            (
                "Capture",
                "updated_img0912.png",
                (
                    "updated_img0912",
                    "png",
                ),
            ),
            ("Skip", "documentation.html"),
            (
                "Capture",
                "favicon.gif",
                (
                    "favicon",
                    "gif",
                ),
            ),
            ("Skip", "img0912.jpg.tmp"),
            ("Skip", "access.lock"),
        ],
    ),
    6: (
        r"^\s*(.*)\s*$",
        [
            ("Capture", "		The quick brown fox...", ("The quick brown fox...",)),
            ("Capture", "   jumps over the lazy dog.", ("jumps over the lazy dog.",)),
        ],
    ),
    7: (
        r"\s+at\s+[\w.]+?(\w+)\(([\w.]+):(\d+)\)",
        #               (\w+)\(([\w\.]+):(\d+)\)
        [
            ("Skip", "W/dalvikvm( 1553): threadid=1: uncaught exception"),
            ("Skip", "E/( 1553): FATAL EXCEPTION: main"),
            ("Skip", "E/( 1553): java.lang.StringIndexOutOfBoundsException"),
            (
                "Capture",
                "E/( 1553):   at widget.List.makeView(ListView.java:1727)",
                (
                    "makeView",
                    "ListView.java",
                    "1727",
                ),
            ),
            (
                "Capture",
                "E/( 1553):   at widget.List.fillDown(ListView.java:652)",
                (
                    "fillDown",
                    "ListView.java",
                    "652",
                ),
            ),
            (
                "Capture",
                "E/( 1553):   at widget.List.fillFrom(ListView.java:709)",
                (
                    "fillFrom",
                    "ListView.java",
                    "709",
                ),
            ),
        ],
    ),
    8: (
        r"(\w+)://([\w\-\.]+)(?::(\d+))?",
        # (\w+)://([\w\-\.]+)(:(\d+))?
        [
            (
                "Capture",
                "ftp://file_server.com:21/top_secret/life_changing_plans.pdf",
                (
                    "ftp",
                    "file_server.com",
                    "21",
                ),
            ),
            (
                "Capture",
                "https://regexone.com/lesson/introduction#section",
                ("https", "regexone.com", None),
            ),
            (
                "Capture",
                "file://localhost:4040/zip_file",
                (
                    "file",
                    "localhost",
                    "4040",
                ),
            ),
            (
                "Capture",
                "https://s3cur3-server.com:9999/",
                (
                    "https",
                    "s3cur3-server.com",
                    "9999",
                ),
            ),
            ("Capture", "market://search/angry%20birds", ("market", "search", None)),
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
    global match  # Allows debugging outside this function
    match = re.search(reg_expr, test_string)
    if test_type == "Match":
        return bool(match)
    elif test_type == "Skip":
        return match is None
    elif test_type == "Capture":
        return match and match.groups() == capture_groups
    else:
        raise ValueError(f"test_type must be Match/Skip/Capture; {test_type} given.")


if __name__ == "__main__":
    for i, (reg_expr, tests) in exercises.items():
        for test in tests:
            assert check_expression(
                reg_expr, *test
            ), f"Exercise {i}: {test} failed (reg_expr={reg_expr}) {match} {match.groups() if match else ''}."
    print("All tests passed!")
