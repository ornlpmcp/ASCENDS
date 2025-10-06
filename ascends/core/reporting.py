"""Optional reporting: md/html + figures."""

from typing import Any


def generate_report(data: Any, output_format: str) -> None:
    """Generate a report from the data.

    Args:
        data: The data to include in the report.
        output_format: The format of the report (md, html).

    """
    if output_format == "md":
        with open("report.md", "w") as f:
            f.write("# Report\n\n")
            f.write(str(data))
    elif output_format == "html":
        with open("report.html", "w") as f:
            f.write("<html><body><h1>Report</h1><p>")
            f.write(str(data))
            f.write("</p></body></html>")
    else:
        raise ValueError("Unsupported format. Use 'md' or 'html'.")
