import re
import random
import string

# קריאה של סיסמאות חלשות מקובץ
with open("common_passwords.txt", "r", encoding="utf-8") as file:
    weak_passwords = set(line.strip().lower() for line in file if line.strip())

def check_password_strength(password):
    score = 0
    suggestions = []

    if len(password) >= 12:
        score += 3
    elif len(password) >= 8:
        score += 2
    else:
        suggestions.append("Use at least 8 characters")

    if re.search(r'[a-z]', password) and re.search(r'[A-Z]', password):
        score += 1
    else:
        suggestions.append("Use both uppercase and lowercase letters")

    if re.search(r'\d', password):
        score += 1
    else:
        suggestions.append("Include at least one digit")

    if re.search(r'[!@#$%^&*(),.?":{}|<>\\[\\]\\\\/]', password):
        score += 1
    else:
        suggestions.append("Include at least one special character")

    if password.lower() in weak_passwords:
        score -= 2
        suggestions.append("Avoid common passwords")

    if re.search(r'(0123|1234|2345|abcd|qwer|asdf|1111)', password.lower()):
        score -= 1
        suggestions.append("Avoid easy patterns or repeated characters")

    if score <= 2:
        strength = "Weak"
    elif score <= 4:
        strength = "Medium"
    else:
        strength = "Strong"

    return strength, score, suggestions

def generate_strong_password(length=12):
    if length < 8:
        length = 8

    lowercase = random.choice(string.ascii_lowercase)
    uppercase = random.choice(string.ascii_uppercase)
    digit = random.choice(string.digits)
    special = random.choice("!@#$%^&*()_+-=[]{}")

    all_chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}"
    remaining_length = length - 4
    remaining = [random.choice(all_chars) for _ in range(remaining_length)]

    password_list = list(lowercase + uppercase + digit + special) + remaining
    random.shuffle(password_list)

    return ''.join(password_list)
