import tkinter as tk
from tkinter import messagebox
import numpy as np
from fpdf import FPDF

# Calculation function


def abundance_ratio(n1, n2, HQS=0.235, normalization=1.7e-10):
    delta_n = n2 - n1
    ratio = normalization * np.exp(-HQS * delta_n)
    return ratio

# PDF export function


def export_pdf(result_str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Li/H Abundance Ratio Calculation", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, result_str)
    pdf.output("Li_H_abundance_result.pdf")
    messagebox.showinfo(
        "Export", "Results exported to Li_H_abundance_result.pdf")

# GUI function


def calculate():
    try:
        n_H = float(entry_nH.get())
        n_Li = float(entry_nLi.get())
        HQS = float(entry_HQS.get())
        normalization = float(entry_norm.get())
        ratio = abundance_ratio(n_H, n_Li, HQS, normalization)
        result_str = (
            f"Hydrogen recursion depth (n_H): {n_H}\n"
            f"Lithium-7 recursion depth (n_Li): {n_Li}\n"
            f"HQS: {HQS}\n"
            f"Normalization: {normalization}\n"
            f"Predicted Li/H ratio: {ratio:.2e}"
        )
        result_label.config(text=f"Li/H ratio: {ratio:.2e}")
        export_button.config(command=lambda: export_pdf(result_str))
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")


# Tkinter GUI setup
root = tk.Tk()
root.title("Li/H Abundance Ratio Calculator")
root.geometry("400x300")

tk.Label(root, text="Hydrogen recursion depth (n_H):").pack()
entry_nH = tk.Entry(root)
entry_nH.insert(0, "42.717")
entry_nH.pack()

tk.Label(root, text="Lithium-7 recursion depth (n_Li):").pack()
entry_nLi = tk.Entry(root)
entry_nLi.insert(0, "45.0")
entry_nLi.pack()

tk.Label(root, text="HQS:").pack()
entry_HQS = tk.Entry(root)
entry_HQS.insert(0, "0.235")
entry_HQS.pack()

tk.Label(root, text="Normalization:").pack()
entry_norm = tk.Entry(root)
entry_norm.insert(0, "1.7e-10")
entry_norm.pack()

tk.Button(root, text="Calculate", command=calculate).pack(pady=10)
result_label = tk.Label(root, text="")
result_label.pack()

export_button = tk.Button(root, text="Export to PDF")
export_button.pack(pady=5)

root.mainloop()
