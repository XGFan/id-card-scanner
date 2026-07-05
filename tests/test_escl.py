from app import escl

SAMPLE_STATUS = """<?xml version="1.0" encoding="UTF-8"?>
<scan:ScannerStatus xmlns:scan="http://schemas.hp.com/imaging/escl/2011/05/03"
    xmlns:pwg="http://www.pwg.org/schemas/2010/12/sm">
  <pwg:Version>2.63</pwg:Version>
  <pwg:State>Idle</pwg:State>
</scan:ScannerStatus>"""


def test_scan_settings_platen_color_region():
    xml = escl.build_scan_settings(300)
    assert "<pwg:InputSource>Platen</pwg:InputSource>" in xml
    assert "<scan:ColorMode>RGB24</scan:ColorMode>" in xml
    assert "<scan:XResolution>300</scan:XResolution>" in xml
    assert "<pwg:Width>2550</pwg:Width>" in xml
    assert "<pwg:Height>3508</pwg:Height>" in xml


def test_parse_scanner_state():
    assert escl.parse_scanner_state(SAMPLE_STATUS) == "Idle"


def test_job_path_handles_absolute_and_relative():
    assert (
        escl.job_path("http://scanner.local:8080/eSCL/ScanJobs/abc")
        == "/eSCL/ScanJobs/abc"
    )
    assert escl.job_path("/eSCL/ScanJobs/abc") == "/eSCL/ScanJobs/abc"
