import sys
import os
import pytest

# ✅ הוספת נתיב לתיקיית המקור – חובה להרצה תקינה
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from password_checker import check_password_strength, generate_strong_password

def test_generated_password_is_strong():
    for _ in range(10):
        pwd = generate_strong_password()
        strength, score, suggestions = check_password_strength(pwd)
        assert strength == "Strong", f"Generated password was not strong: {pwd}"
        assert score >= 5, f"Score too low: {score} for password: {pwd}"
        assert len(pwd) >= 8, f"Password too short: {pwd}"
        assert any(c.islower() for c in pwd), "Missing lowercase letter"
        assert any(c.isupper() for c in pwd), "Missing uppercase letter"
        assert any(c.isdigit() for c in pwd), "Missing digit"
        assert any(c in "!@#$%^&*()_+-=[]{}" for c in pwd), "Missing special character"

def test_empty_password():
    strength, score, suggestions = check_password_strength("")
    assert strength == "Weak"
    assert score <= 1
    assert "Use at least 8 characters" in suggestions

def test_common_password():
    strength, score, suggestions = check_password_strength("123456")
    assert strength == "Weak"
    assert score <= 1
    assert any("common" in s.lower() or "avoid" in s.lower() for s in suggestions)

def test_medium_password():
    strength, score, suggestions = check_password_strength("Pass123!")
    assert strength == "Medium" or strength == "Strong"


def test_strong_password():
    strength, score, suggestions = check_password_strength("T0ugh#Passw0rd!")
    assert strength == "Strong"
    assert score >= 6
    assert suggestions == []

def test_pattern_password():
    strength, score, suggestions = check_password_strength("qwerty123")
    assert strength == "Weak"
    assert any("pattern" in s.lower() or "simple" in s.lower() or "sequence" in s.lower() for s in suggestions)
