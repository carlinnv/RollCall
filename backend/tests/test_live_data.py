"""DATA_SOURCE=live: the app starts empty and students and absences come from SIS CSV imports."""
from fastapi.testclient import TestClient

from app.main import app
from app.store import Store, data_source

HEADER = ("student_id,first_name,last_name,grade,classroom,teacher,bus_route,household_id,"
          "preferred_language,guardian_name,guardian_phone,date,sis_code\n")


def test_demo_by_default_and_in_demo_mode(monkeypatch):
    assert data_source() == "demo"
    monkeypatch.setenv("DATA_SOURCE", "live")
    assert data_source() == "live"
    monkeypatch.setenv("DEMO_MODE", "1")
    assert data_source() == "demo"  # the scripted demo always runs on the synthetic school


def test_live_starts_empty(monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "live")
    s = Store()
    assert s.students == {} and s.absences == {}


def test_import_creates_and_updates_students(monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "live")
    s = Store()
    csv = HEADER + (
        "T001,Ana,Lopez,3,Room 3B,Ms. Park,Route 2,H01,es,Maria Lopez,+1-555-0201,2026-11-10,unexcused\n"
        "T001,Ana,Lopez,3,Room 3B,Ms. Park,Route 2,H01,es,Maria Lopez,+1-555-0201,2026-11-11,unexcused\n"
        "T002,Ben,Ng,4,Room 4A,Mr. Diaz,,H02,en,Kim Ng,+1-555-0202,2026-11-10,excused\n"
        "T003,Cy,,4,Room 4A,Mr. Diaz,,H03,en,Lee Cy,+1-555-0203,2026-11-10,excused\n"  # no last name
    )
    added, errors, students = s.import_csv(csv)
    assert (added, students) == (3, 2)
    assert len(errors) == 1 and "last_name" in errors[0]
    assert s.students["T001"].preferred_language == "es"
    assert s.students["T002"].bus_route is None  # empty column: no bus

    # A later export updates the student and skips absences already logged.
    again = HEADER + "T001,Ana,Lopez,3,Room 3C,Ms. Park,Route 2,H01,es,Maria Lopez,+1-555-0201,2026-11-10,unexcused\n"
    assert s.import_csv(again) == (0, [], 0)
    assert s.students["T001"].classroom == "Room 3C"


def test_import_without_student_columns_needs_known_students(monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "live")
    added, errors, _ = Store().import_csv("student_id,date,sis_code\nT404,2026-11-10,unexcused\n")
    assert added == 0 and "unknown student_id" in errors[0]


def test_reset_refused_with_live_data(monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "live")
    r = TestClient(app).post("/api/reset")
    assert r.status_code == 403
    assert TestClient(app).get("/api/health").json()["data_source"] == "live"


def test_roster_rows_add_students_without_absences(monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "live")
    s = Store()
    added, errors, students = s.import_csv(HEADER + "T009,Ivy,Ross,2,Room 2C,Ms. Hwang,,H09,en,Jo Ross,+1-555-0209,,\n")
    assert (added, errors, students) == (0, [], 1)
    assert "T009" in s.students and not s.absences


def test_sample_csv_imports_cleanly(monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("DATA_SOURCE", "live")
    s = Store()
    text = (Path(__file__).resolve().parents[1] / "data" / "sample" / "sis_import.csv").read_text()
    added, errors, students = s.import_csv(text)
    assert errors == [] and added == 28 and students == 24
