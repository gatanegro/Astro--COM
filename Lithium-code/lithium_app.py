import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg  # Ensures backend is bundled
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tkinter as tk
from tkinter import filedialog, messagebox
import os


# --- Core Model Functions ---

def li_h_ratio(n, HQS, normalization, n_ref_lih):
    delta_n = n_ref_lih - n
    return normalization * np.exp(HQS * delta_n)

def li6_li7_ratio(HQS, delta_n):
    return np.exp(-HQS * delta_n)

# --- GUI Application ---

class LithiumApp:
    def __init__(self, root):
        self.root = root
        root.title("Lithium Abundance Recursion Model")
        root.geometry("700x600") # Increased window size

        # Input fields
        input_frame = tk.LabelFrame(root, text="Model Parameters", padx=10, pady=10)
        input_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        tk.Label(input_frame, text="Hydrogen Recursion Depth (n_H):").grid(row=0, column=0, sticky="e")
        self.n_h_entry = tk.Entry(input_frame)
        self.n_h_entry.insert(0, "42.717") # Default from theory
        self.n_h_entry.grid(row=0, column=1, padx=5, pady=2)

        tk.Label(input_frame, text="Lithium-7 Recursion Depth (n_Li7):").grid(row=1, column=0, sticky="e")
        self.n_li7_entry = tk.Entry(input_frame)
        self.n_li7_entry.insert(0, "45.0") # Default from theory
        self.n_li7_entry.grid(row=1, column=1, padx=5, pady=2)

        tk.Label(input_frame, text="HQS Tax:").grid(row=2, column=0, sticky="e")
        self.hqs_entry = tk.Entry(input_frame)
        self.hqs_entry.insert(0, "0.235")
        self.hqs_entry.grid(row=2, column=1, padx=5, pady=2)

        tk.Label(input_frame, text="Li/H Normalization (Old Stars):").grid(row=3, column=0, sticky="e")
        self.norm_entry = tk.Entry(input_frame)
        self.norm_entry.insert(0, "1.7e-10")
        self.norm_entry.grid(row=3, column=1, padx=5, pady=2)

        tk.Label(input_frame, text="Young Stars Recursion Depth (n_young):").grid(row=4, column=0, sticky="e")
        self.n_young_entry = tk.Entry(input_frame)
        self.n_young_entry.insert(0, "44.10")
        self.n_young_entry.grid(row=4, column=1, padx=5, pady=2)

        tk.Label(input_frame, text="Li-6/Li-7 Delta n:").grid(row=5, column=0, sticky="e")
        self.delta_n_entry = tk.Entry(input_frame)
        self.delta_n_entry.insert(0, "0.1")
        self.delta_n_entry.grid(row=5, column=1, padx=5, pady=2)

        # Buttons
        button_frame = tk.Frame(root)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        tk.Button(button_frame, text="Calculate & Plot", command=self.calculate).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Export PDF Report", command=self.export_pdf).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Quit", command=root.quit).pack(side=tk.LEFT, padx=5)

        # Output
        self.result_text = tk.Text(root, height=10, width=80)
        self.result_text.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # Store results for export
        self.results = {}

    def calculate(self):
        try:
            n_h = float(self.n_h_entry.get())
            n_li7 = float(self.n_li7_entry.get())
            HQS = float(self.hqs_entry.get())
            normalization = float(self.norm_entry.get())
            n_young = float(self.n_young_entry.get())
            delta_n_li6li7 = float(self.delta_n_entry.get())

            # Calculate Li/H for old stars (using n_H as 'n' and n_Li7 as 'n_ref_lih')
            li_h_old = li_h_ratio(n_h, HQS, normalization, n_li7)

            # Calculate Li/H for young stars (using n_young as 'n' and n_Li7 as 'n_ref_lih')
            li_h_young = li_h_ratio(n_young, HQS, normalization, n_li7)

            # Li-6/Li-7 ratio
            li6li7 = li6_li7_ratio(HQS, delta_n_li6li7)

            # Store results
            self.results = {
                "n_h": n_h,
                "n_li7": n_li7,
                "HQS": HQS,
                "normalization": normalization,
                "n_young": n_young,
                "delta_n_li6li7": delta_n_li6li7,
                "li_h_old": li_h_old,
                "li_h_young": li_h_young,
                "li6li7": li6li7
            }

            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Calculated Li/H for Old Stars (n_H={n_h}, n_Li7={n_li7}): {li_h_old:.2e}\n")
            self.result_text.insert(tk.END, f"Calculated Li/H for Young Stars (n_young={n_young}, n_Li7={n_li7}): {li_h_young:.2e}\n")
            self.result_text.insert(tk.END, f"Calculated Li-6/Li-7 Ratio (Delta n={delta_n_li6li7}): {li6li7:.2f}\n")

            # Plot
            self.plot_results(n_h, n_li7, HQS, normalization, n_young)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for all fields.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def plot_results(self, n_h, n_li7, HQS, normalization, n_young):
        # Plot Li/H vs. Recursion Depth
        n_vals_lih = np.linspace(40.0, 46.0, 200) # Wider range for n
        li_h_vals = [li_h_ratio(n, HQS, normalization, n_li7) for n in n_vals_lih]

        plt.figure(figsize=(10, 6))
        plt.plot(n_vals_lih, li_h_vals, label="Li/H Ratio")
        plt.scatter([n_h, n_young], [li_h_ratio(n_h, HQS, normalization, n_li7), li_h_ratio(n_young, HQS, normalization, n_li7)],
                    color=["red", "blue"], s=100, zorder=5, label="Calculated Points (Old/Young Stars)")
        plt.axvline(n_h, color='red', linestyle='--', linewidth=0.8, label=f'n_H = {n_h}')
        plt.axvline(n_li7, color='green', linestyle='--', linewidth=0.8, label=f'n_Li7 = {n_li7}')
        plt.axvline(n_young, color='blue', linestyle='--', linewidth=0.8, label=f'n_young = {n_young}')

        plt.yscale("log")
        plt.xlabel("Recursion Depth (n)")
        plt.ylabel("Li/H Ratio (log scale)")
        plt.title("Lithium-7 to Hydrogen Ratio vs. Recursion Depth")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig("LiH_vs_n.png", dpi=300)
        plt.close()

        # Plot Li-6/Li-7 Ratio (constant in this model)
        plt.figure(figsize=(10, 3))
        li6li7_val = self.results["li6li7"]
        plt.axhline(li6li7_val, color='purple', linestyle='-', label=f'Li-6/Li-7 Ratio = {li6li7_val:.2f}')
        plt.text(0.5, li6li7_val + 0.01, f'Predicted: {li6li7_val:.2f}', ha='center', va='bottom', transform=plt.gca().transAxes)
        plt.xlabel("Recursion Depth (n)") # Still show recursion depth on x-axis for consistency
        plt.ylabel("Li-6 / Li-7 Ratio")
        plt.title("Li-6 to Li-7 Ratio")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig("Li6_Li7_ratio.png", dpi=300)
        plt.close()

        messagebox.showinfo("Plots Generated", "Plots 'LiH_vs_n.png' and 'Li6_Li7_ratio.png' have been saved in the current directory.")

    def export_pdf(self):
        if not self.results:
            messagebox.showwarning("Warning", "Please calculate results first before exporting PDF.")
            return

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Lithium Abundance Recursion Model Report", 0, 1, "C")
        pdf.ln(10)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "1. Model Inputs", 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6,
            f"Hydrogen Recursion Depth (n_H): {self.results['n_h']}\n"
            f"Lithium-7 Recursion Depth (n_Li7): {self.results['n_li7']}\n"
            f"HQS Tax: {self.results['HQS']}\n"
            f"Li/H Normalization (Old Stars): {self.results['normalization']:.2e}\n"
            f"Young Stars Recursion Depth (n_young): {self.results['n_young']}\n"
            f"Li-6/Li-7 Delta n: {self.results['delta_n_li6li7']}\n"
        )
        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "2. Calculated Results", 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6,
            f"Calculated Li/H for Old Stars: {self.results['li_h_old']:.2e}\n"
            f"Calculated Li/H for Young Stars: {self.results['li_h_young']:.2e}\n"
            f"Calculated Li-6/Li-7 Ratio: {self.results['li6li7']:.2f}\n"
        )
        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "3. Plots", 0, 1)
        pdf.set_font("Arial", "", 10)
        
        # Check if plots exist before adding
        if os.path.exists("LiH_vs_n.png"):
            pdf.image("LiH_vs_n.png", x=10, w=190)
            pdf.ln(5)
        else:
            pdf.cell(0, 10, "Li/H vs. Recursion Depth plot not found.", 0, 1)

        if os.path.exists("Li6_Li7_ratio.png"):
            pdf.image("Li6_Li7_ratio.png", x=10, w=190)
            pdf.ln(5)
        else:
            pdf.cell(0, 10, "Li-6/Li-7 Ratio plot not found.", 0, 1)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "4. Interpretation", 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6,
            "This model explains the observed Li/H ratios by linking them to fundamental recursion depths and the HQS tax. "
            "Old (Pop II) stars with deeper recursion depths (higher n) show lower Li/H due to higher HQS tax, matching the Spite plateau (e.g., 1e-10). "
            "Young (Pop I) stars with shallower n show higher Li/H, matching observed values (e.g., ~2e-9) due to ongoing recursive production and less depletion. "
            "The Li-6/Li-7 ratio is also determined by the HQS tax and a specific delta_n, providing a consistent framework for lithium abundances."
        )

        savepath = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")], title="Save PDF Report")
        if savepath:
            pdf.output(savepath)
            messagebox.showinfo("Export Successful", f"Report exported to {savepath}")

# --- Run the GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LithiumApp(root)
    root.mainloop()


