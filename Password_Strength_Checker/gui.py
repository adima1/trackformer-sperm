import tkinter as tk
import random
import string
from tkinter import messagebox
from password_checker import check_password_strength,generate_strong_password  # הקובץ שלך

show_password = False  # משתנה גלובלי למעקב


def insert_generated_password():
    generated = generate_strong_password()
    entry.delete(0, tk.END)
    entry.insert(0, generated)
    evaluate_password()


def evaluate_password():
    password = entry.get()
    if not password:
        messagebox.showwarning("Warning", "Please enter a password.")
        return

    strength, score, suggestions = check_password_strength(password)

    result_label.config(text=f"Strength: {strength} ({score}/7)", fg=color_for(strength))

    if suggestions:
        suggestion_text = "\n".join(f"• {s}" for s in suggestions)
    else:
        suggestion_text = "✅ Good job! Your password is strong."
    suggestions_label.config(text=suggestion_text)

def color_for(strength):
    return {
        "Weak": "#e74c3c",    # אדום
        "Medium": "#f39c12",  # כתום
        "Strong": "#27ae60"   # ירוק
    }.get(strength, "black")

def toggle_password():
    global show_password
    show_password = not show_password
    entry.config(show="" if show_password else "*")
    toggle_btn.config(text="🙈" if show_password else "👁️")

# יצירת חלון
root = tk.Tk()
root.title("Password Strength Checker")
root.geometry("460x360")
root.configure(bg="#f4f4f4")
root.resizable(False, False)

# 🔒 אייקון לחלון (אם קיים קובץ ico)
try:
    root.iconbitmap("lock.ico")  # שים lock.ico באותה תיקייה
except:
    pass  # אם אין, פשוט ממשיך בלי אייקון

# כותרת
tk.Label(root, text="🔐 Check Your Password Strength", bg="#f4f4f4",
         font=("Helvetica", 14, "bold")).pack(pady=15)

# שדה קלט עם כפתור הצגה/הסתרה
entry_frame = tk.Frame(root, bg="#f4f4f4")
entry_frame.pack()

entry = tk.Entry(entry_frame, width=30, show="*", font=("Helvetica", 12))
entry.grid(row=0, column=0, padx=(0, 5))

toggle_btn = tk.Button(entry_frame, text="👁️", command=toggle_password, font=("Helvetica", 10),
                       bg="white", relief="flat", cursor="hand2")
toggle_btn.grid(row=0, column=1)

# כפתור בדיקה
tk.Button(root, text="Evaluate", command=evaluate_password, font=("Helvetica", 12),
          bg="#3498db", fg="white", padx=10, pady=5).pack(pady=10)

tk.Button(root, text="Generate Strong Password", command=insert_generated_password,
          font=("Helvetica", 10), bg="#2ecc71", fg="white", padx=8, pady=4).pack(pady=(0, 10))


# מסגרת תוצאה
frame = tk.Frame(root, bg="white", bd=2, relief="groove")
frame.pack(padx=20, pady=10, fill="both", expand=False)

result_label = tk.Label(frame, text="", font=("Helvetica", 14, "bold"), bg="white")
result_label.pack(pady=8)

suggestions_label = tk.Label(frame, text="", font=("Helvetica", 10), bg="white", justify="left", wraplength=400)
suggestions_label.pack(pady=4, padx=10)

# הפעלת החלון
root.mainloop()
